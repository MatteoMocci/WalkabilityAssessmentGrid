"""Late-fusion helpers for combining two single-view classifiers."""

# combined_utils.py
from typing import Optional
import torch
import torch.nn as nn

class CombinedLateFusion(nn.Module):
    """
    Late-fusion model: two single-view classifiers -> concat logits -> small linear head.
    Forward expects named args: street=..., sat=...
    """
    def __init__(self, street_model: nn.Module, sat_model: nn.Module, num_classes: int):
        super().__init__()
        self.street_model = street_model
        self.sat_model = sat_model
        self.comb_head = nn.Linear(num_classes * 2, num_classes)

    @staticmethod
    def _to_logits(out):
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        return out

    def forward(self, *, street: torch.Tensor, sat: torch.Tensor) -> torch.Tensor:
        ls = self._to_logits(self.street_model(street))
        la = self._to_logits(self.sat_model(sat))
        return self.comb_head(torch.cat([ls, la], dim=1))


def train_two_single_heads_one_epoch(
    street_model: nn.Module,
    sat_model: nn.Module,
    loader,                     # dual-view loader: returns (street, sat), label
    device: torch.device,
    criterion: nn.Module,
    street_opt: torch.optim.Optimizer,
    sat_opt: torch.optim.Optimizer,
    logger=None,
    clip_grad: Optional[float] = 5.0,
):
    """
    Trains the classification heads of the two single-view models for one epoch,
    using the SAME dual-view batches so samples are aligned across branches.
    """
    street_model.train()
    sat_model.train()

    for step, (imgs, labels) in enumerate(loader):
        if not (isinstance(imgs, (tuple, list)) and len(imgs) == 2):
            raise RuntimeError("Combined training expects a dual-view loader (returns (street, sat), label).")
        s, sa = imgs
        s = s.to(device, non_blocking=True)
        sa = sa.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- street branch ---
        street_opt.zero_grad(set_to_none=True)
        out_s = street_model(s)
        if hasattr(out_s, "logits"): out_s = out_s.logits
        elif isinstance(out_s, dict) and "logits" in out_s: out_s = out_s["logits"]
        loss_s = criterion(out_s, labels)
        loss_s.backward()
        if clip_grad: nn.utils.clip_grad_norm_(street_model.parameters(), max_norm=clip_grad)
        street_opt.step()

        # --- satellite branch ---
        sat_opt.zero_grad(set_to_none=True)
        out_a = sat_model(sa)
        if hasattr(out_a, "logits"): out_a = out_a.logits
        elif isinstance(out_a, dict) and "logits" in out_a: out_a = out_a["logits"]
        loss_a = criterion(out_a, labels)
        loss_a.backward()
        if clip_grad: nn.utils.clip_grad_norm_(sat_model.parameters(), max_norm=clip_grad)
        sat_opt.step()

        if logger and (step % 50 == 0):
            logger.info(f"[combined] step {step:04d} loss_street={loss_s.item():.4f} loss_sat={loss_a.item():.4f}")
