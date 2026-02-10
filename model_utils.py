"""Model loading and classifier-head replacement utilities for torchvision/timm backbones."""

import importlib
import os
import warnings
import torch
import torch.nn as nn
from torchvision import models
import timm

try:
    certifi = importlib.import_module("certifi")
except Exception:
    certifi = None

try:
    CvtForImageClassification = getattr(
        importlib.import_module("transformers"), "CvtForImageClassification"
    )
except Exception:
    CvtForImageClassification = None

# --- trust Windows cert store for HTTPS (fixes corporate proxy/self-signed roots) ---
if certifi is not None:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
# ------------------------------------------------------------------------------------

def _load_cvt_hf(num_classes: int, variant: str = "microsoft/cvt-13"):
    """
    Load a CvT model from HuggingFace with a resized classifier head.
    """
    if CvtForImageClassification is None:
        raise RuntimeError("Transformers is required for CvT. pip install transformers")
    # create model with a head sized for your labels; ignore_mismatched_sizes lets us replace the classifier cleanly
    return CvtForImageClassification.from_pretrained(
        variant, num_labels=num_classes, ignore_mismatched_sizes=True
    )

# --- timm alias resolver: hand timm canonical names, not short aliases ---
_TIMM_ALIAS_CANDIDATES = {
    "vit_base":  ["vit_base_patch16_224"],
    "deit_base": ["deit_base_patch16_224"],
    "swin_base": ["swin_base_patch4_window7_224"],
    "beit_base": ["beit_base_patch16_224"],
    # CvT names in timm vary by version; try a few
    "cvt_base":  ["cvt-21", "cvt-21-384", "cvt-13"],
}

# Map timm logical keys to expected HF filenames, so we can load locally if provided
_TIMM_HF_FILES = {
    "swin_base": "swin_base_patch4_window7_224.ms_in22k_ft_in1k.safetensors",
    "beit_base": "beit_base_patch16_224.in22k_ft_in22k_in1k.safetensors",
}

def _maybe_local_checkpoint(key: str) -> str | None:
    """
    If LOCAL_WEIGHTS_DIR is set and contains a known .safetensors for this model key,
    return its full path so we don't hit the network.
    """
    root = os.environ.get("LOCAL_WEIGHTS_DIR")
    if not root:
        return None
    fname = _TIMM_HF_FILES.get(key)
    if not fname:
        return None
    path = os.path.join(root, fname)
    return path if os.path.exists(path) else None

def _create_timm_by_alias(key: str, pretrained: bool, drop: float):
    """
    Resolve 'key' to a canonical timm name the current timm version supports.
    Prefer local checkpoint if provided. Otherwise, try pretrained; on any download
    failure (SSL/offline), fall back to pretrained=False so the run continues.
    """
    if timm is None:
        return None

    # list once to avoid repeated calls; support older timm that may not accept both flags
    try:
        all_names = set(timm.list_models(pretrained=True)) | set(timm.list_models(pretrained=False))
    except Exception:
        all_names = set(timm.list_models())

    cands = _TIMM_ALIAS_CANDIDATES.get(key, [key])

    # local checkpoint first, if available
    local_ckpt = _maybe_local_checkpoint(key)

    for name in cands:
        if name not in all_names:
            continue

        if local_ckpt:
            warnings.warn(f"Using local checkpoint for '{key}': {local_ckpt}")
            return timm.create_model(
                name,
                pretrained=False,
                drop_rate=drop,
                drop_path_rate=drop,
                checkpoint_path=local_ckpt,
            )

        # try normal pretrained path
        try:
            return timm.create_model(
                name,
                pretrained=bool(pretrained),
                drop_rate=drop,
                drop_path_rate=drop,
            )
        except RuntimeError as e:
            # Unknown model -> try next candidate; other errors -> fall back to random init
            if "Unknown model" in str(e):
                continue
            warnings.warn(f"Pretrained load failed for '{name}' ({e}); using pretrained=False.")
            return timm.create_model(
                name,
                pretrained=False,
                drop_rate=drop,
                drop_path_rate=drop,
            )
        except Exception as e:
            warnings.warn(f"Pretrained load failed for '{name}' ({e}); using pretrained=False.")
            return timm.create_model(
                name,
                pretrained=False,
                drop_rate=drop,
                drop_path_rate=drop,
            )

    # no candidate matched this timm version
    return None

HEAD_DROPOUT = 0.3
TF_DROPOUT = 0.1

def _disable_googlenet_aux(m):
    """
    Disable GoogLeNet auxiliary heads so forward returns only main logits.
    """
    # ensure forward returns only main logits
    if hasattr(m, "aux_logits"):
        m.aux_logits = False
    if hasattr(m, "aux1"):
        m.aux1 = None
    if hasattr(m, "aux2"):
        m.aux2 = None
    return m

# model factories for ImageNet1k pretraining (timm + torchvision)
_MODEL_FACTORIES = {
    "alexnet":    lambda pretrained: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1) if pretrained else models.alexnet(num_classes=1000),
    "vgg16":      lambda pretrained: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)     if pretrained else models.vgg16(num_classes=1000),
    "googlenet":  lambda pretrained: (_disable_googlenet_aux(models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1))
                                      if pretrained else models.googlenet(aux_logits=False)),
    "resnet50":   lambda pretrained: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) if pretrained else models.resnet50(num_classes=1000),

    # timm transformers via canonical names
    "vit_base":   lambda pretrained: _create_timm_by_alias("vit_base",  pretrained, TF_DROPOUT),
    "deit_base":  lambda pretrained: _create_timm_by_alias("deit_base", pretrained, TF_DROPOUT),
    "swin_base":  lambda pretrained: _create_timm_by_alias("swin_base", pretrained, TF_DROPOUT),
    "beit_base":  lambda pretrained: _create_timm_by_alias("beit_base", pretrained, TF_DROPOUT),
    "cvt_base":   lambda pretrained: _create_timm_by_alias("cvt_base",  pretrained, TF_DROPOUT),
}

def replace_head(model, num_classes: int):
    """
    Replace the classifier head of a backbone with a new num_classes head.

    Steps:
    1) Detect model family (timm vs torchvision).
    2) Replace the appropriate classifier layer.
    3) Keep the new head on the model's device.
    """
    # keep the new layer on the same device as existing params
    def _dev(m):
        """Return the device of the first parameter, or CPU if none."""
        try:
            return next(m.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    dev = _dev(model)

    # 1) timm models (Swin, ViT, BEiT, DeiT, ConvNeXt, etc.)
    if hasattr(model, "reset_classifier"):
        try:
            model.reset_classifier(num_classes=num_classes)
            return model.to(dev)
        except Exception:
            pass  # fall back to manual cases

    # 2) torchvision CNNs
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):        # ResNet/GoogLeNet
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes).to(dev)
        return model

    if hasattr(model, "classifier"):                                    # VGG/AlexNet
        if isinstance(model.classifier, nn.Sequential):
            seq = list(model.classifier)
            idxs = [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
            if idxs:
                last = idxs[-1]
                in_f = seq[last].in_features
                seq[last] = nn.Linear(in_f, num_classes).to(dev)
                model.classifier = nn.Sequential(*seq)
                return model

    # 3) common timm pattern: .head is Linear or Sequential
    if hasattr(model, "head"):
        head = model.head
        if isinstance(head, nn.Linear):
            in_f = head.in_features
            model.head = nn.Linear(in_f, num_classes).to(dev)
            return model
        if isinstance(head, nn.Sequential):
            seq = list(head)
            idxs = [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
            if idxs:
                last = idxs[-1]
                in_f = seq[last].in_features
                seq[last] = nn.Linear(in_f, num_classes).to(dev)
                model.head = nn.Sequential(*seq)
                return model

    # 4) DeiT distillation head (timm) — if present, align it too
    if hasattr(model, "head_dist") and isinstance(model.head_dist, nn.Linear):
        in_f = model.head_dist.in_features
        model.head_dist = nn.Linear(in_f, num_classes).to(dev)
        return model

    raise TypeError(f"replace_head: unsupported model type {type(model)}")

def load_model(name: str, pretrained_source: str, num_classes: int | None):
    """
    Load a backbone model and optionally replace its classifier head.

    Steps:
    1) Validate pretraining source and model name.
    2) Load the pretrained backbone.
    3) Replace the head if num_classes is provided.
    """
    key = name.lower()

    if pretrained_source.lower() != "imagenet1k":
        # hard stop so nothing tries Places365 anymore
        raise RuntimeError(f"SKIP: pretraining '{pretrained_source}' disabled")

    if key not in _MODEL_FACTORIES:
        raise ValueError(f"Unknown model name: {name}")

    # Try to get a pretrained model; if the factory returns None, this timm version lacks it
    model = _MODEL_FACTORIES[key](True)
    if model is None:
        raise RuntimeError(f"SKIP: timm model '{key}' not available in this timm version")

    if num_classes is not None:
        model = replace_head(model, num_classes)
    return model

def is_cnn_model(model: nn.Module) -> bool:
    """
    Return True if the model looks like a CNN (vs transformer).
    """
    name = type(model).__name__.lower()
    return any(k in name for k in ("resnet", "alexnet", "vgg", "googlenet"))
