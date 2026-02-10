"""Convert Places365 VGG16 Caffe weights to a PyTorch-usable checkpoint."""

# tools/convert_places_vgg16.py
# Run inside the Python 3.10 env after installing:
#   pip install onnx==1.6.0 caffe2onnx==2.0.1 onnx2pytorch==0.4.1
import importlib
import torch
import pathlib, subprocess, sys, urllib.request
import onnx
from onnx2pytorch import ConvertModel

BASE = pathlib.Path(__file__).resolve().parents[1]
W = BASE / "weights"
W.mkdir(exist_ok=True)

PROTO_URL = "http://places2.csail.mit.edu/models_places365/vgg16_places365.prototxt"
CAFFE_URL = "http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel"

PROTO = W / "vgg16_places365.prototxt"
CAFFE = W / "vgg16_places365.caffemodel"
ONNXP = W / "vgg16_places365.onnx"
PTP   = W / "vgg16_places365.pth"

def dl(url, out):
    """
    Download a file only if it does not already exist.

    Steps:
    1) Check output path.
    2) Fetch from URL if missing.
    """
    if not out.exists():
        print("download ->", out)
        urllib.request.urlretrieve(url, out)

def main():
    """
    Convert Places365 VGG16 Caffe weights into a PyTorch checkpoint.

    Steps:
    1) Verify caffe2onnx is installed.
    2) Download prototxt and caffemodel if needed.
    3) Convert Caffe -> ONNX -> PyTorch.
    4) Save the PyTorch model.
    """
    # make sure caffe2onnx is importable
    try:
        importlib.import_module("caffe2onnx")
    except Exception as e:
        print("Please install caffe2onnx, onnx==1.6.0 and onnx2pytorch in a Python 3.10 env.")
        print("Example:")
        print("  py -3.10 -m venv .venv-310 && .\\.venv-310\\Scripts\\activate")
        print("  pip install onnx==1.6.0 caffe2onnx==2.0.1 onnx2pytorch==0.4.1")
        raise

    dl(PROTO_URL, PROTO)
    dl(CAFFE_URL, CAFFE)

    if not ONNXP.exists():
        print("convert caffe -> onnx ...")
        # use the caffe2onnx CLI (works without a Caffe install)
        subprocess.check_call([
            sys.executable, "-m", "caffe2onnx.convert",
            "--prototxt", str(PROTO),
            "--caffemodel", str(CAFFE),
            "--onnx", str(ONNXP)
        ])

    print("load onnx and convert to pytorch ...")
    onnx_model = onnx.load(str(ONNXP))
    torch_model = ConvertModel(onnx_model)  # nn.Module
    torch_model.eval()
    torch.save(torch_model, PTP)
    print("saved ->", PTP)

if __name__ == "__main__":
    main()
