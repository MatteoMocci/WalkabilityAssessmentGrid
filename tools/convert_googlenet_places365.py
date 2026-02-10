"""Convert Places365 GoogLeNet Caffe weights to torchvision backbone state dict."""

# tools/convert_googlenet_places365.py
# Converts Caffe GoogLeNet (Places365) weights to a torchvision GoogLeNet backbone state dict.
# Output: weights/googlenet_places365_backbone.pth  (conv-only; no classifier)
#
# Requirements:
#   - protoc on PATH (protoc --version should work)
#   - Python packages already in your env: protobuf, torch, torchvision
#   - Vendored single-file helper (already in your repo from earlier steps):
#       third_party/caffemodel2pytorch.py   (we only use its 'initialize' to compile caffe.proto)

import importlib.util
from pathlib import Path
import numpy as np
import torch
import torchvision
from google.protobuf import text_format

def _load_caffemodel2pytorch_initialize():
    """
    Load the vendored Caffe-to-PyTorch helper and return its initialize() function.

    Steps:
    1) Resolve the helper path in third_party/.
    2) Import the module dynamically.
    3) Return the initialize function.
    """
    helper = Path(__file__).resolve().parents[1] / "third_party" / "caffemodel2pytorch.py"
    spec = importlib.util.spec_from_file_location("caffemodel2pytorch", helper)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module at {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.initialize

initialize = _load_caffemodel2pytorch_initialize()

BASE = Path(__file__).resolve().parents[1]
W    = BASE / "weights"
W.mkdir(exist_ok=True)

PROTO = W / "googlenet_places365.prototxt"
CAFFE = W / "googlenet_places365.caffemodel"
OUT   = W / "googlenet_places365_backbone.pth"

CAFFE_PROTO = "https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto"

def _blob_to_array(blob):
    """
    Convert a Caffe BlobProto to a NumPy array with a best-effort shape.

    Steps:
    1) Read dimensions from blob.shape or legacy fields.
    2) Convert raw data to float32 NumPy.
    3) Reshape if dimensions match; otherwise return flat/2D data.
    """
    # Robustly convert Caffe BlobProto to np.ndarray
    if hasattr(blob, "shape") and getattr(blob.shape, "dim", None):
        dims = list(blob.shape.dim)
    else:
        # older fields
        dims = [d for d in (blob.num, blob.channels, blob.height, blob.width) if d > 0]
    arr = np.asarray(blob.data, dtype=np.float32)
    if dims and np.prod(dims) == arr.size:
        return arr.reshape(dims)
    return arr  # fallback (e.g., FC weights may already be 2D)

def _load_caffe_params(prototxt_path, caffemodel_path):
    """
    Load Caffe weights into a dict mapping layer name -> (weights, bias).

    Steps:
    1) Initialize Caffe protobuf definitions.
    2) Parse prototxt for sanity (optional).
    3) Parse caffemodel blobs into NumPy arrays.
    """
    caffe_pb2 = initialize(CAFFE_PROTO)
    # parse prototxt (optional, but helpful for sanity)
    net_param = caffe_pb2.NetParameter()
    with open(prototxt_path, "r", encoding="utf-8") as f:
        text_format.Merge(f.read(), net_param)

    # parse caffemodel (weights)
    weights = caffe_pb2.NetParameter()
    with open(caffemodel_path, "rb") as f:
        weights.ParseFromString(f.read())

    # layer list can be 'layer' or legacy 'layers'
    layers = list(getattr(weights, "layer", [])) or list(getattr(weights, "layers", []))
    blobs = {}
    for L in layers:
        if not L.blobs:
            continue
        w = _blob_to_array(L.blobs[0])
        b = _blob_to_array(L.blobs[1]) if len(L.blobs) > 1 else None
        blobs[L.name] = (w, b)
    return blobs

def _set_basicconv2d(basic, w, b):
    """
    Assign weights/bias to a torchvision BasicConv2d layer.

    Steps:
    1) Copy convolution weights.
    2) Copy bias if present.
    """
    # basic is torchvision.models.googlenet.BasicConv2d
    basic.conv.weight.data.copy_(torch.from_numpy(w))
    if basic.conv.bias is not None and b is not None:
        basic.conv.bias.data.copy_(torch.from_numpy(b))

def _set_conv(conv, w, b):
    """
    Assign weights/bias to a convolution layer.

    Steps:
    1) Copy convolution weights.
    2) Copy bias if present.
    """
    conv.weight.data.copy_(torch.from_numpy(w))
    if conv.bias is not None and b is not None:
        conv.bias.data.copy_(torch.from_numpy(b))

def convert():
    """
    Convert Places365 GoogLeNet Caffe weights into a torchvision backbone checkpoint.

    Steps:
    1) Load Caffe blobs.
    2) Build a torchvision GoogLeNet.
    3) Map blobs into each inception block.
    4) Save backbone-only state dict.
    """
    if not PROTO.exists() or not CAFFE.exists():
        raise FileNotFoundError(
            "Missing weights/googlenet_places365.prototxt or .caffemodel. "
            "Place the files under weights/ before running this script."
        )

    print("Loading Caffe weights ...")
    caffe = _load_caffe_params(PROTO, CAFFE)
    print(f"Loaded blobs: {len(caffe)} layers with params")

    print("Building torchvision GoogLeNet ...")
    model = torchvision.models.googlenet(weights=None, aux_logits=False)  # no aux heads

    # 1) stem
    _set_conv(model.conv1.conv, *caffe["conv1/7x7_s2"])
    _set_conv(model.conv2.conv, *caffe["conv2/3x3_reduce"])
    _set_conv(model.conv3.conv, *caffe["conv2/3x3"])

    # 2) inception blocks mapping helper
    def map_inception(prefix: str, block):
        """
        Map Caffe inception weights into a torchvision inception block.

        Steps:
        1) Load 1x1 branch.
        2) Load 3x3 reduce + 3x3 branch.
        3) Load 5x5 reduce + 5x5 branch.
        4) Load pool projection branch.
        """
        # branch1
        _set_basicconv2d(block.branch1, *caffe[f"{prefix}/1x1"])
        # branch2: reduce -> 3x3
        _set_basicconv2d(block.branch2[0], *caffe[f"{prefix}/3x3_reduce"])
        _set_basicconv2d(block.branch2[1], *caffe[f"{prefix}/3x3"])
        # branch3: reduce -> 5x5
        _set_basicconv2d(block.branch3[0], *caffe[f"{prefix}/5x5_reduce"])
        _set_basicconv2d(block.branch3[1], *caffe[f"{prefix}/5x5"])
        # branch4: pool_proj
        _set_basicconv2d(block.branch4[1], *caffe[f"{prefix}/pool_proj"])

    # 3) all inception blocks
    map_inception("inception_3a", model.inception3a)
    map_inception("inception_3b", model.inception3b)
    map_inception("inception_4a", model.inception4a)
    map_inception("inception_4b", model.inception4b)
    map_inception("inception_4c", model.inception4c)
    map_inception("inception_4d", model.inception4d)
    map_inception("inception_4e", model.inception4e)
    map_inception("inception_5a", model.inception5a)
    map_inception("inception_5b", model.inception5b)

    # We intentionally DO NOT load the final classifier (loss3/classifier) or aux heads.
    # You will train a new Linear(1024, num_classes) on top of the backbone.

    # Save only the backbone params (everything except model.fc)
    sd = model.state_dict()
    sd = {k: v.cpu() for k, v in sd.items() if not k.startswith("fc.")}
    torch.save(sd, OUT)
    print(f"Saved backbone state dict -> {OUT}")

if __name__ == "__main__":
    convert()
