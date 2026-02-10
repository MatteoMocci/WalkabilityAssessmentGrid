"""WMS satellite cutout downloader for heading-aware geospatial sampling."""

# wms_batch_cutouts.py
# requirements:
#   pip install requests pillow pyproj lxml pandas

import re
import math
import requests
import pandas as pd
from io import BytesIO
from pathlib import Path
from typing import Tuple

from PIL import Image
from pyproj import Transformer
from lxml import etree

# -----------------------------
# CONFIG — set your mode here
# -----------------------------
CSV_PATH = "input/points.csv"  # must contain columns: lat, lon, heading
OUTPUT_DIR = "wms_cutouts"

MODE = "single"        # "single" or "pair_by_offset"
GROUND_WIDTH_M = 60            # ground window size (width = height), meters
OUT_PX = 600                   # final output size for each tile (600x600)
FETCH_PX = 900                 # fetch a bit larger to be safe for crops

# for MODE = "pair_by_offset"
OFFSET_M = 15                  # shift of the second image center, meters (perpendicular to heading)
MERGE_PAIR_TO_PANO = True      # also save a 1200x600 pano made from A|B

# WMS candidates (2019 first, then 2013)
CAPABILITIES_CANDIDATES = [
    "https://map.sitr.regione.sicilia.it/gis/services/ortofoto/ortofoto_2019_20cm_sicilia/ImageServer/WMSServer?request=GetCapabilities&service=WMS",
    "https://map.sitr.regione.sicilia.it/gis/services/ortofoto/ortofoto_2013_15cm_comuni/MapServer/WMSServer?request=GetCapabilities&service=WMS",
]

LAYER_HINT = "orto"            # bias layer auto pick


# -----------------------------
# small helpers
# -----------------------------
def fetch_capabilities() -> Tuple[str, etree._Element]:
    """
    Try each WMS GetCapabilities endpoint and return the first that succeeds.

    Steps:
    1) Iterate candidate URLs.
    2) Fetch and parse XML.
    3) Raise if all fail.
    """
    last_err = None
    for url in CAPABILITIES_CANDIDATES:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return url, etree.fromstring(r.content)
        except Exception as e:
            last_err = e
    raise SystemExit(f"Could not fetch WMS Capabilities. Last error: {last_err}")

def pick_layer_and_crs(xml: etree._Element) -> Tuple[str, str, str]:
    """
    Choose the best imagery layer and supported CRS from WMS capabilities.

    Steps:
    1) Extract named layers.
    2) Score layers by keywords and CRS support.
    3) Pick the top-scoring layer and a preferred CRS.
    """
    ns = {"wms": "http://www.opengis.net/wms", "xlink": "http://www.w3.org/1999/xlink"}
    version = xml.get("version", "1.3.0")
    is_130 = version.startswith("1.3")

    layers = xml.xpath("//wms:Capability//wms:Layer[wms:Name]", namespaces=ns)
    if not layers:
        raise SystemExit("No named layers found in this WMS capabilities.")

    def text(el, qname):
        """Get stripped text for a qualified XML element, or empty string."""
        node = el.find(qname, namespaces=ns)
        return node.text.strip() if node is not None and node.text else ""

    def supports_crs(el, code: str) -> bool:
        """Return True if the layer advertises support for a CRS/SRS code."""
        tag = "CRS" if is_130 else "SRS"
        for crs in el.findall(f"wms:{tag}", namespaces=ns):
            if crs.text and code.upper() in crs.text.upper():
                return True
        return False

    def score_layer(el) -> float:
        """
        Score a layer using keyword hints and CRS support.
        Higher score indicates a better imagery candidate.
        """
        name = text(el, "wms:Name")
        title = text(el, "wms:Title")
        abstract = text(el, "wms:Abstract")
        desc = " ".join([name, title, abstract]).lower()
        hint = LAYER_HINT.lower().strip()

        score = 0
        if re.search(r"\b(orto|orthophoto|ortho|imagery|raster|foto|aereo|rgb)\b", desc):
            score += 5
        if hint and hint in desc:
            score += 3
        if supports_crs(el, "EPSG:3857"):
            score += 3
        if supports_crs(el, "EPSG:4326"):
            score += 1
        m = re.search(r"(20\d{2})", desc)
        if m:
            score += (int(m.group(1)) - 2000) / 100.0
        return score

    best = sorted(layers, key=score_layer, reverse=True)[0]
    layer_name = text(best, "wms:Name")

    crs_tag = "CRS" if is_130 else "SRS"
    supported = [crs.text.strip() for crs in best.findall(f"wms:{crs_tag}", namespaces=ns) if crs is not None and crs.text]
    for candidate in ["EPSG:3857", "EPSG:32633", "EPSG:32632", "EPSG:4326"]:
        if any(candidate in s for s in supported):
            chosen_crs = candidate
            break
    else:
        chosen_crs = supported[0] if supported else "EPSG:3857"

    return version, layer_name, chosen_crs

def build_bbox(lat: float, lon: float, crs: str, ground_width_m: float) -> Tuple[float,float,float,float]:
    """
    Build a square bounding box around a center in a target CRS.

    Steps:
    1) Convert to target CRS if needed.
    2) Compute half-width in CRS units.
    3) Return minx, miny, maxx, maxy.
    """
    to_target = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    if crs == "EPSG:4326":
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
        half_w_deg = (ground_width_m / meters_per_deg_lon) / 2.0
        half_h_deg = (ground_width_m / meters_per_deg_lat) / 2.0
        minx, miny = lon - half_w_deg, lat - half_h_deg
        maxx, maxy = lon + half_w_deg, lat + half_h_deg
    else:
        x, y = to_target.transform(lon, lat)
        half = ground_width_m / 2.0
        minx, miny, maxx, maxy = x - half, y - half, x + half, y + half
    return minx, miny, maxx, maxy

def getmap_png(base_capabilities_url: str, version: str, layer: str, crs: str,
               bbox: Tuple[float,float,float,float], width: int, height: int) -> Image.Image:
    """
    Request a WMS GetMap PNG for a bbox and return it as a PIL image.

    Steps:
    1) Build request parameters (CRS/SRS and bbox ordering).
    2) Execute HTTP request.
    3) Raise on XML error responses.
    """
    is_130 = version.startswith("1.3")
    base = base_capabilities_url.split("?")[0]
    params = {
        "SERVICE": "WMS",
        "VERSION": version,
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/png",
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "TRANSPARENT": "FALSE",
    }
    minx, miny, maxx, maxy = bbox
    if is_130:
        params["CRS"] = crs
        if crs == "EPSG:4326":
            params["BBOX"] = f"{miny},{minx},{maxy},{maxx}"  # lat,lon in 1.3.0
        else:
            params["BBOX"] = f"{minx},{miny},{maxx},{maxy}"
    else:
        params["SRS"] = crs
        params["BBOX"] = f"{minx},{miny},{maxx},{maxy}"

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    if r.headers.get("Content-Type", "").startswith("text/xml"):
        raise SystemExit(f"WMS returned XML error:\n{r.text[:1000]}")
    return Image.open(BytesIO(r.content))

def center_crop(img: Image.Image, size: int) -> Image.Image:
    """
    Center-crop a square region from an image.
    """
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))

def offset_point(lat: float, lon: float, heading_deg: float, dx_m: float, dy_m: float) -> Tuple[float,float]:
    """
    Offset (lat, lon) by dx, dy meters in heading-aligned local frame.
    dx is forward along heading, dy is left of heading. Positive dy for heading+90.
    """
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    th = math.radians(heading_deg)
    de_east = dx_m * math.sin(th) + dy_m * math.cos(th)
    de_north = dx_m * math.cos(th) - dy_m * math.sin(th)
    dlon = de_east / meters_per_deg_lon
    dlat = de_north / meters_per_deg_lat
    return lat + dlat, lon + dlon


# -----------------------------
# main
# -----------------------------
def main():
    """
    CLI entry point to download WMS cutouts for each CSV point.

    Steps:
    1) Fetch WMS capabilities and select layer/CRS.
    2) Read input CSV and iterate points.
    3) Download, crop, and save cutouts (and optional pano).
    """
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching WMS capabilities...")
    cap_url, xml = fetch_capabilities()
    version, layer, crs = pick_layer_and_crs(xml)
    print(f"WMS: {cap_url}")
    print(f"Layer: {layer}  |  CRS: {crs}")

    df = pd.read_csv(CSV_PATH)
    colmap = {c.lower(): c for c in df.columns}
    for need in ("lat", "lon", "heading"):
        if need not in colmap:
            raise SystemExit(f"CSV must contain column: {need}")
    LAT, LON, HEADING = colmap["lat"], colmap["lon"], colmap["heading"]

    for i, row in df.iterrows():
        lat = float(row[LAT]); lon = float(row[LON])
        try: heading = float(row[HEADING])
        except Exception: heading = 0.0

        # centers: A = original, B = perpendicular offset (heading+90)
        centers = [(lat, lon)]
        if MODE == "pair_by_offset":
            lat2, lon2 = offset_point(lat, lon, heading, dx_m=0.0, dy_m=OFFSET_M)
            centers.append((lat2, lon2))

        tiles = []
        names = []
        for j, (clat, clon) in enumerate(centers):
            bbox = build_bbox(clat, clon, crs, GROUND_WIDTH_M)
            big = getmap_png(cap_url, version, layer, crs, bbox, FETCH_PX, FETCH_PX)
            img = center_crop(big, OUT_PX)
            tag = "a" if j == 0 else "b"
            fname = f"{clon:.6f}_{clat:.6f}_{int(round(heading))}_{tag}.png"
            img.save(out_dir / fname)
            print(f"[{i}] Saved {fname}")
            tiles.append(img)
            names.append(fname)

        # optional 180° pano: simple horizontal concat A|B (1200x600)
        if MODE == "pair_by_offset" and MERGE_PAIR_TO_PANO and len(tiles) == 2:
            pano = Image.new("RGB", (OUT_PX * 2, OUT_PX))
            pano.paste(tiles[0], (0, 0))
            pano.paste(tiles[1], (OUT_PX, 0))
            pano_name = f"{lon:.6f}_{lat:.6f}_{int(round(heading))}_pano.png"
            pano.save(out_dir / pano_name)
            print(f"[{i}] Saved {pano_name} (180° pano)")

    print("Done.")

if __name__ == "__main__":
    main()
