"""Metric helpers for multi-class walkability evaluation."""

# metrics_utils.py
import numpy as np
from typing import Dict, Iterable, Optional, Any

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

def _as_labels(x: Iterable) -> np.ndarray:
    """
    Convert input iterable to a 1D int64 NumPy array of labels.

    Steps:
    1) Convert to NumPy.
    2) Flatten to 1D if needed.
    3) Cast to int64.
    """
    a = np.asarray(list(x))
    if a.ndim != 1:
        a = a.reshape(-1)
    # cast to int
    return a.astype(np.int64, copy=False)

def compute_metrics(
    preds: Iterable,
    refs: Iterable,
    num_classes: int = 5,
    **kwargs: Any,  # swallow any unexpected kwargs from old call sites
) -> Dict[str, float]:
    """
    Pure sklearn implementation:
      - accuracy (micro)
      - precision/recall/f1 (macro, zero_division=0)
      - mean_absolute_error
      - one_off_accuracy (|pred-true| <= 1)

    Args:
        preds: 1D iterable of predicted class ids
        refs:  1D iterable of ground truth class ids
        num_classes: number of classes (default 5)

    Returns:
        dict with keys: accuracy, precision, recall, f1, mean_absolute_error, one_off_accuracy
    """
    y_pred = _as_labels(preds)
    y_true = _as_labels(refs)

    if y_pred.size == 0 or y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_absolute_error": float("nan"),
            "one_off_accuracy": float("nan"),
        }

    # sanity: clamp to valid range if needed (optional)
    if num_classes is not None and num_classes > 0:
        y_pred = np.clip(y_pred, 0, num_classes - 1)
        y_true = np.clip(y_true, 0, num_classes - 1)

    acc = float(accuracy_score(y_true, y_pred))

    # Macro-averaged, robust to missing predicted/true classes
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)),
        average="macro", zero_division=0
    )
    prec = float(prec); rec = float(rec); f1 = float(f1)

    # Ordinal-friendly extras
    mae = float(np.mean(np.abs(y_pred - y_true)))
    one_off = float(np.mean(np.abs(y_pred - y_true) <= 1))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_absolute_error": mae,
        "one_off_accuracy": one_off,
    }

def compute_accuracy(preds: Iterable, refs: Iterable) -> float:
    """Kept for backward compatibility with older call sites."""
    y_pred = _as_labels(preds)
    y_true = _as_labels(refs)
    if y_pred.size == 0 or y_true.size == 0:
        return 0.0
    return float(accuracy_score(y_true, y_pred))

def confusion(y_true: Iterable, y_pred: Iterable, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Compute a confusion matrix with optional fixed class labels.

    Steps:
    1) Normalize inputs to label arrays.
    2) Build label list if num_classes provided.
    3) Return sklearn confusion matrix.
    """
    y_pred = _as_labels(y_pred)
    y_true = _as_labels(y_true)
    labels = list(range(num_classes)) if num_classes is not None else None
    return confusion_matrix(y_true, y_pred, labels=labels)
