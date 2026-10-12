#!/usr/bin/env python3
"""Create the Python environment for this repository on Linux, Windows or macOS.

    python setup_env.py                 # recommended: .venv, versions and PyTorch build chosen automatically
    python setup_env.py --exact         # exact versions of the published results (needs Python 3.12)
    python setup_env.py --torch cpu     # force a PyTorch build: auto, cpu, pypi, cu118 ... cu130

Steps:
1) Create the virtual environment (.venv) with the running Python (3.10 or newer).
2) Choose the package versions: requirements-lock.txt (exact, Python 3.12) or
   requirements.txt (compatible ranges, any supported Python).
3) Install torch/torchvision from the PyTorch index that matches the NVIDIA driver
   (newest CUDA build the driver supports; falls back to older builds, then CPU).
   macOS uses the default PyPI build (Apple GPU through MPS when available).
4) Install the remaining packages and check that everything imports and which device
   training will use.
Only the standard library is used, so this runs before anything is installed.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
TORCH_INDEX = "https://download.pytorch.org/whl/{}"
CUDA_BUILDS = [("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6)), ("cu124", (12, 4)),
               ("cu121", (12, 1)), ("cu118", (11, 8))]
LOCK_PYTHON = (3, 12)

CHECK = r"""
import importlib.util, platform, torch, torchvision, timm, albumentations, cv2, numpy, scipy, sklearn, PIL, tqdm
if torch.cuda.is_available():
    device = f"cuda ({torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda})"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps (Apple GPU)"
else:
    device = "cpu (training will be very slow)"
compiled = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
print(f"  python {platform.python_version()}, torch {torch.__version__}, torchvision {torchvision.__version__}, "
      f"timm {timm.__version__}, albumentations {albumentations.__version__}, numpy {numpy.__version__}")
print(f"  training device: {device}")
print(f"  torch.compile: {'available' if compiled else 'not available here, models run uncompiled (slower)'}")
"""


def venv_python(path):
    """Interpreter inside a virtual environment, on any OS."""
    return os.path.join(path, "Scripts", "python.exe") if os.name == "nt" else os.path.join(path, "bin", "python")


def driver_cuda():
    """
    Highest CUDA version the NVIDIA driver supports, or None without an NVIDIA GPU.
    Asks the driver library directly (works even when nvidia-smi is not on PATH, e.g. in
    sandboxed terminals); falls back to parsing nvidia-smi.
    """
    import ctypes
    import ctypes.util
    names = ["nvcuda.dll"] if os.name == "nt" else ["libcuda.so.1", "libcuda.so", ctypes.util.find_library("cuda")]
    for name in filter(None, names):
        try:
            lib = ctypes.WinDLL(name) if os.name == "nt" else ctypes.CDLL(name)
            version = ctypes.c_int()
            if lib.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value > 0:
                return version.value // 1000, (version.value % 1000) // 10
        except (OSError, AttributeError):
            continue
    candidates = [shutil.which("nvidia-smi"), "/usr/bin/nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi",
                  r"C:\Windows\System32\nvidia-smi.exe",
                  r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]
    for exe in filter(None, candidates):
        if not os.path.exists(exe):
            continue
        try:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def installed_torch_build(py):
    """Local build tag of the torch already in the environment ('cpu', 'cu130', '' for PyPI), or None."""
    res = subprocess.run([py, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True)
    if res.returncode != 0:
        return None
    version = res.stdout.strip()
    return version.split("+", 1)[1] if "+" in version else ""


def torch_sources(choice):
    """Ordered list of (label, build tag, pip index args) to try for torch/torchvision."""
    if choice == "pypi" or (choice == "auto" and platform.system() == "Darwin"):
        return [("PyPI default build", "", [])]
    if choice != "auto":
        return [(f"{choice} build", choice, ["--index-url", TORCH_INDEX.format(choice)])]
    cuda = driver_cuda()
    if cuda is None:
        print("No NVIDIA GPU driver found: installing the CPU build of PyTorch "
              "(use --torch cu126 or similar to override).")
        return [("CPU build", "cpu", ["--index-url", TORCH_INDEX.format("cpu")])]
    print(f"NVIDIA driver supports CUDA up to {cuda[0]}.{cuda[1]}.")
    builds = [(f"{tag} build", tag, ["--index-url", TORCH_INDEX.format(tag)])
              for tag, need in CUDA_BUILDS if need <= cuda]
    return builds + [("CPU build", "cpu", ["--index-url", TORCH_INDEX.format("cpu")])]


def pip(py, *args):
    return subprocess.run([py, "-m", "pip", *args]).returncode == 0


def torch_pins(req_file):
    """torch / torchvision requirement lines of a requirements file."""
    with open(req_file, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    return [l for l in lines if re.match(r"^(torch|torchvision)\b", l)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", default=os.path.join(HERE, ".venv"), help="environment folder (default .venv)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--exact", action="store_true", help="exact versions of the published results (Python 3.12)")
    mode.add_argument("--compatible", action="store_true", help="compatible version ranges (any Python >= 3.10)")
    ap.add_argument("--torch", default="auto", help="PyTorch build: auto (default), cpu, pypi, or a CUDA tag like cu126")
    args = ap.parse_args()

    if sys.version_info < (3, 10):
        raise SystemExit(f"Python 3.10 or newer is needed (this is {platform.python_version()}).")
    exact = args.exact or (not args.compatible and sys.version_info[:2] == LOCK_PYTHON)
    if args.exact and sys.version_info[:2] != LOCK_PYTHON:
        raise SystemExit("--exact needs Python 3.12: the pinned numpy/scipy versions do not exist for "
                         f"Python {platform.python_version()}. Use Python 3.12, or drop --exact.")
    req = os.path.join(HERE, "requirements-lock.txt" if exact else "requirements.txt")
    print(f"Python {platform.python_version()} on {platform.system()}: "
          f"{'exact versions of the published results' if exact else 'compatible versions'} ({os.path.basename(req)})")
    if not exact:
        print("  (results can differ slightly from the published ones; use Python 3.12 for the exact environment)")

    if not os.path.exists(venv_python(args.venv)):
        print(f"Creating the environment in {args.venv}")
        venv.EnvBuilder(with_pip=True).create(args.venv)
    py = venv_python(args.venv)
    pip(py, "install", "--upgrade", "pip")

    pins = torch_pins(req)
    installed = None
    for label, tag, index in torch_sources(args.torch):
        current = installed_torch_build(py)
        if current is not None and current != tag:
            # pip would keep a torch of the right version but the wrong build (e.g. CPU instead of CUDA)
            print(f"Replacing the installed PyTorch build ({current or 'PyPI'}) with the {label}")
            pip(py, "uninstall", "-y", "torch", "torchvision")
        print(f"Installing PyTorch ({label}): {' '.join(pins)}")
        if pip(py, "install", *pins, *index):
            installed = label
            break
        if exact:
            continue  # exact versions not published for this build: try the next one
        print(f"  not available as {label} with these versions, trying the newest release of this build")
        if pip(py, "install", "torch", "torchvision", *index):
            installed = label
            break
    if installed is None:
        raise SystemExit("Could not install PyTorch. Install it by hand from https://pytorch.org/get-started/ "
                         "and run this script again with --torch pypi.")

    print(f"Installing the other packages from {os.path.basename(req)}")
    if not pip(py, "install", "-r", req):
        raise SystemExit("Package installation failed (see the pip messages above).")

    print("Checking the environment:")
    if subprocess.run([py, "-c", CHECK]).returncode != 0:
        raise SystemExit("The environment does not import correctly (see the error above).")
    run = venv_python(os.path.relpath(args.venv, HERE))
    print(f"\nDone. Run the pipeline with this interpreter, e.g.:\n  {run} study.py status\n"
          f"  {run} study.py grid\n(see README for all commands)")


if __name__ == "__main__":
    main()
