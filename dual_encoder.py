"""Dual-view and combined-view feature-fusion model definitions."""

# dual_encoder.py
import torch
import torch.nn as nn

def _flatten_if_needed(x):
    """
    Flatten spatial feature maps by global average if needed.

    Steps:
    1) Check for [B, C, H, W] tensors.
    2) Average over H and W to produce [B, C].
    """
    # Handles [B, C, H, W] -> [B, C] by global average if needed
    if x.dim() == 4:
        x = x.mean(dim=(2, 3))
    return x

# ---------------------------
# Feature extractors (generic)
# ---------------------------

class _FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int):
        """
        Wrap a backbone and expose a standardized feature extractor.
        """
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim

    def forward(self, x):
        """
        Forward pass that returns extracted features.
        """
        return self._features(self.backbone, x)

    # --- per-arch feature extraction ---
    @staticmethod
    def _features(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from different backbone families.

        Steps:
        1) Handle timm models via forward_features.
        2) Handle torchvision CNNs (ResNet/AlexNet/VGG/GoogLeNet).
        3) Fallback to model output/logits.
        """
        name = type(m).__name__.lower()

        # timm-style models
        if hasattr(m, "forward_features"):
            feats = m.forward_features(x)
            # try to get pre-logits if available
            if hasattr(m, "forward_head"):
                try:
                    feats = m.forward_head(feats, pre_logits=True)
                except TypeError:
                    # older timm without pre_logits kw
                    feats = m.forward_head(feats)
            feats = _flatten_if_needed(feats)
            return feats

        # torchvision ResNet
        if "resnet" in name and hasattr(m, "layer4"):
            x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
            x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
            if hasattr(m, "avgpool"):
                x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        # torchvision AlexNet / VGG
        if "alexnet" in name or "vgg" in name:
            x = m.features(x)
            if hasattr(m, "avgpool"):
                x = m.avgpool(x)
            x = torch.flatten(x, 1)
            # take penultimate fc representation
            if hasattr(m, "classifier"):
                # try all but last layer
                seq = list(m.classifier.children())
                if len(seq) > 1:
                    head = nn.Sequential(*seq[:-1])
                    x = head(x)
            return x

        # torchvision GoogLeNet
        if "googlenet" in name:
            # minimal feature path (no aux heads)
            x = m.conv1(x); x = m.maxpool1(x)
            x = m.conv2(x); x = m.conv3(x); x = m.maxpool2(x)
            x = m.inception3a(x); x = m.inception3b(x); x = m.maxpool3(x)
            x = m.inception4a(x); x = m.inception4b(x); x = m.inception4c(x)
            x = m.inception4d(x); x = m.inception4e(x); x = m.maxpool4(x)
            x = m.inception5a(x); x = m.inception5b(x)
            if hasattr(m, "avgpool"):
                x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        # Fallback: run full model and take logits as "features"
        out = m(x)
        if isinstance(out, dict) and "logits" in out:
            out = out["logits"]
        return out

    @staticmethod
    def feature_dim(backbone: nn.Module) -> int:
        """
        Infer feature dimension for a backbone model.

        Steps:
        1) Use model metadata (num_features/fc.in_features).
        2) Fallback to a dummy forward pass.
        """
        # timm common attribute
        if hasattr(backbone, "num_features"):
            return int(backbone.num_features)

        name = type(backbone).__name__.lower()
        if "resnet" in name and hasattr(backbone, "fc"):
            return int(backbone.fc.in_features)
        if "alexnet" in name or "vgg" in name:
            # penultimate classifier width
            if hasattr(backbone, "classifier"):
                seq = [m for m in backbone.classifier if isinstance(m, nn.Linear)]
                if len(seq) >= 2:
                    return int(seq[-2].out_features)
            return 4096
        if "googlenet" in name and hasattr(backbone, "fc"):
            return int(backbone.fc.in_features)

        # last resort: infer by a small dummy forward
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                f = _FeatureExtractor._features(backbone, dummy)
                return int(f.shape[-1])
        except Exception:
            return 1024


def make_feature_extractor(backbone: nn.Module) -> _FeatureExtractor:
    """
    Create a feature extractor wrapper for a backbone.
    """
    return _FeatureExtractor(backbone, _FeatureExtractor.feature_dim(backbone))

class DualFeatureConcat(nn.Module):
    """
    Feature-level fusion for dual view using generic extractors:
      f_street ⊕ f_sat -> 512 -> ReLU -> Dropout(0.3) -> logits
    """
    def __init__(self, street_backbone: nn.Module, sat_backbone: nn.Module, num_classes: int):
        """
        Initialize dual-encoder feature concatenation head.
        """
        super().__init__()
        self.st_ex = make_feature_extractor(street_backbone)
        self.sa_ex = make_feature_extractor(sat_backbone)

        f_st = self.st_ex.feat_dim
        f_sa = self.sa_ex.feat_dim

        self.head = nn.Sequential(
            nn.Linear(f_st + f_sa, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.is_pair_input = True

    def forward(self, street, sat):
        """
        Forward pass: extract features from both views and classify.
        """
        fs = self.st_ex(street)
        fa = self.sa_ex(sat)
        x = torch.cat([fs, fa], dim=1)
        return self.head(x)


class CombinedFeatureConcat(torch.nn.Module):
    """
    Late fusion via feature concatenation + MLP head:
      features(street) ⊕ features(sat) -> 512 -> ReLU -> Dropout(0.3) -> logits
    Uses generic feature extractors so it works with torchvision/timm backbones.
    """
    def __init__(self, street_model: nn.Module, sat_model: nn.Module, num_classes: int):
        """
        Initialize late-fusion feature concatenation head.
        """
        super().__init__()
        self.st = street_model
        self.sa = sat_model
        self.st_ex = make_feature_extractor(self.st)
        self.sa_ex = make_feature_extractor(self.sa)

        f_st = self.st_ex.feat_dim
        f_sa = self.sa_ex.feat_dim
        self.head = nn.Sequential(
            nn.Linear(f_st + f_sa, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.is_pair_input = True  # so routing uses (street, sat)

    def forward(self, street: torch.Tensor, sat: torch.Tensor):
        """
        Forward pass: extract features, concatenate, and classify.
        """
        fs = self.st_ex(street)
        fa = self.sa_ex(sat)
        x = torch.cat([fs, fa], dim=1)
        return self.head(x)


class CombinedLogitsFusion(torch.nn.Module):
    def __init__(self, street_model, sat_model, mode: str = "avg"):
        """
        Initialize logits-level fusion for two single-view models.

        Steps:
        1) Store models.
        2) Set fusion mode (avg/sum/learned).
        """
        super().__init__()
        self.st = street_model
        self.sa = sat_model
        self.mode = mode
        self.is_pair_input = True
        if mode == "learned":
            self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        else:
            self.register_parameter("alpha", None)

    def forward(self, street, sat):
        """
        Forward pass: fuse logits from two models.

        Steps:
        1) Compute street and satellite logits.
        2) Combine according to fusion mode.
        """
        ls = self.st(street)
        la = self.sa(sat)
        if self.mode == "avg":
            return 0.5 * (ls + la)
        if self.mode == "sum":
            return ls + la
        if self.mode == "learned":
            a = torch.sigmoid(self.alpha)
            return a * ls + (1.0 - a) * la
        raise ValueError(f"Unknown fusion mode {self.mode}")

