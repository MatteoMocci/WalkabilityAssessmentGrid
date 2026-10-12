"""Build spatial cross-validation folds for the labelled survey points.

Points are grouped into square spatial blocks (default 1 km) and whole blocks
are assigned to folds with StratifiedGroupKFold, stratifying on city x mean
rating so every fold sees a similar mix of cities and scores. No point is dropped.

The fold table is written once and reused by every experiment, so all models
are trained and tested on exactly the same splits.

Usage:
    python make_folds.py --manifest "<Satellite Extraction>/manifest.json"
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

EARTH_RADIUS_M = 6371000.0
REF_LAT_DEG = 40.0  # fixed reference latitude for the local metric projection


def project_m(lon: float, lat: float):
    """
    Project lon/lat to a local equirectangular metric plane (metres).
    Accurate to well under 1% over Sardinia, enough to define 1 km blocks.
    """
    x = EARTH_RADIUS_M * math.radians(lon) * math.cos(math.radians(REF_LAT_DEG))
    y = EARTH_RADIUS_M * math.radians(lat)
    return x, y


def load_labels(path: str):
    """
    Read point_labels.csv and return {point_id: mean_rating}.
    The mean rating is 1 + sum_k p_above_k.
    """
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["point_id"]] = 1.0 + sum(float(r[f"p_above_{k}"]) for k in range(1, 5))
    return out


def load_manifest_points(path: str):
    """
    Read the survey manifest and return {point_id: (city, lon, lat)}.
    """
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    out = {}
    for entries in manifest.values():
        for e in entries:
            out[e["point_id"]] = (e["city"], float(e["lon"]), float(e["lat"]))
    return out


def cross_fold_neighbour_share(xy: np.ndarray, folds: np.ndarray, radius_m: float) -> float:
    """
    Share of points that have at least one point from another fold within radius_m.
    """
    order = np.argsort(xy[:, 0])
    xs = xy[order, 0]
    flagged = np.zeros(len(xy), dtype=bool)
    for a in range(len(order)):
        b = a + 1
        while b < len(order) and xs[b] - xs[a] <= radius_m:
            i, j = order[a], order[b]
            if folds[i] != folds[j] and np.hypot(*(xy[i] - xy[j])) <= radius_m:
                flagged[i] = flagged[j] = True
            b += 1
    return float(flagged.mean())


def main():
    """
    CLI entry point.

    Steps:
    1) Join labels with survey coordinates.
    2) Assign each point to a spatial block.
    3) Split blocks into folds stratified by mean rating and write the table.
    """
    ap = argparse.ArgumentParser(description="Build spatial-block CV folds")
    ap.add_argument("--labels", default="point_labels.csv")
    ap.add_argument("--manifest", required=True, help="Survey manifest.json with point coordinates")
    ap.add_argument("--out", default="folds.csv")
    ap.add_argument("--block-m", type=float, default=1000.0)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify", choices=["city_rating", "rating"], default="city_rating")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    coords = load_manifest_points(args.manifest)
    missing = sorted(set(labels) - set(coords))
    if missing:
        raise SystemExit(f"{len(missing)} labelled points have no coordinates, e.g. {missing[:5]}")

    ids = sorted(labels)
    xy = np.array([project_m(coords[p][1], coords[p][2]) for p in ids])
    blocks = [f"{int(x // args.block_m)}_{int(y // args.block_m)}" for x, y in xy]
    # strata: the 9 possible mean ratings (1.0, 1.5, ..., 5.0), optionally crossed with city
    cities = sorted({coords[p][0] for p in ids})
    strata = np.array([int(round(2 * labels[p])) for p in ids])
    if args.stratify == "city_rating":
        strata = strata + 100 * np.array([cities.index(coords[p][0]) for p in ids])

    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = np.full(len(ids), -1)
    for k, (_, test_idx) in enumerate(sgkf.split(xy, strata, groups=blocks)):
        folds[test_idx] = k
    assert (folds >= 0).all()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point_id", "city", "lon", "lat", "block", "mean_rating", "fold"])
        for i, p in enumerate(ids):
            city, lon, lat = coords[p]
            w.writerow([p, city, f"{lon:.6f}", f"{lat:.6f}", blocks[i], f"{labels[p]:.1f}", int(folds[i])])

    # report
    print(f"Wrote {args.out}: {len(ids)} points, {len(set(blocks))} blocks of {args.block_m:.0f} m, {args.n_folds} folds")
    by_fold = defaultdict(list)
    for i in range(len(ids)):
        by_fold[folds[i]].append(i)
    for k in range(args.n_folds):
        idx = by_fold[k]
        mr = [labels[ids[i]] for i in idx]
        cities = Counter(coords[ids[i]][0] for i in idx)
        print(f"  fold {k}: n={len(idx):4d}  blocks={len({blocks[i] for i in idx}):3d}  "
              f"mean rating={np.mean(mr):.2f}  " + " ".join(f"{c}={n}" for c, n in sorted(cities.items())))
    for r in (75, 150):
        print(f"  points with a point from another fold within {r} m: "
              f"{100 * cross_fold_neighbour_share(xy, folds, r):.1f}%")


if __name__ == "__main__":
    main()
