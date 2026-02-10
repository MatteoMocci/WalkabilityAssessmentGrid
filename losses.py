"""Loss definitions and factory helpers (CE, weighted CE, and SCE)."""


import torch
import torch.nn as nn
import torch.nn.functional as F

class SCELoss(nn.Module):
    def __init__(self, alpha: float, beta: float, num_classes: int):
        """
        Initialize Symmetric Cross Entropy loss hyperparameters.
        """
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor):
        """
        Compute SCE = alpha * CE + beta * RCE.

        Steps:
        1) Compute standard cross entropy.
        2) Compute reverse cross entropy on one-hot labels.
        3) Combine the two terms.
        """
        ce = self.cross_entropy(pred, labels)

        # DO NOT use clamp_ on tensors that require grad
        prob = F.softmax(pred, dim=1)
        prob = torch.clamp(prob, min=1e-7, max=1.0)

        # one_hot does not require grad, but avoid in place here too
        one_hot = F.one_hot(labels, self.num_classes).to(pred.device).float()
        one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)

        rce = -torch.sum(prob * torch.log(one_hot), dim=1).mean()
        return self.alpha * ce + self.beta * rce


def get_criterion(name: str, num_classes: int, class_weights: torch.Tensor | None = None) -> nn.Module:
    """
    Factory for loss functions (CE, WCE, SCE).

    Steps:
    1) Normalize the name.
    2) Build the requested loss (with weights if needed).
    3) Validate unsupported options.
    """
    key = name.upper()
    if key == "CE":
        return nn.CrossEntropyLoss()
    elif key == "WCE":
        if class_weights is None:
            raise ValueError("WCE requires class_weights")
        return nn.CrossEntropyLoss(weight=class_weights)
    elif key == "SCE":
        return SCELoss(alpha=0.5, beta=1.0, num_classes=num_classes)
    else:
        raise ValueError("loss must be one of: CE, WCE, SCE")
