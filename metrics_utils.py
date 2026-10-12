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


# ====================================================================
# Ordinal walkability metrics (two raters per point, 1-5 scale)
# ====================================================================

def cumulative_from_output(outputs: np.ndarray, head: str) -> np.ndarray:
    """
    Convert raw model outputs to cumulative probabilities P(Y > k), k = 1..4.

    - ordinal head: 4 logits -> sigmoid
    - ce head: 5 logits -> softmax -> P(Y > k) = sum_{j > k} p_j
    """
    outputs = np.asarray(outputs, dtype=np.float64)
    if head == "ordinal":
        return 1.0 / (1.0 + np.exp(-outputs))
    if head == "ce":
        z = outputs - outputs.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return np.cumsum(p[:, ::-1], axis=1)[:, ::-1][:, 1:]
    raise ValueError(f"unknown head {head}")


def soft_confusion(pred: np.ndarray, rater_lo: np.ndarray, rater_hi: np.ndarray,
                   num_classes: int = 5) -> np.ndarray:
    """
    5x5 confusion matrix (rows = rating, cols = prediction, classes 1..5) where
    each point counts 0.5 toward each of its two raters' ratings.
    """
    cm = np.zeros((num_classes, num_classes))
    np.add.at(cm, (rater_lo - 1, pred - 1), 0.5)
    np.add.at(cm, (rater_hi - 1, pred - 1), 0.5)
    return cm


def _class_probs(cum: np.ndarray) -> np.ndarray:
    """Class probabilities P(Y = j), j = 1..5, from cumulative P(Y > k) (clipped if non-monotone)."""
    c = np.concatenate([np.ones((len(cum), 1)), cum, np.zeros((len(cum), 1))], axis=1)
    p = np.clip(c[:, :-1] - c[:, 1:], 0.0, None)
    return p / p.sum(axis=1, keepdims=True)


def _metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    """
    Discrete metrics from a (possibly fractional) confusion matrix, rows = rating, cols = prediction:
    accuracy, macro-F1, MAE, one-off, macro-averaged MAE (AMAE), maximum per-class MAE (MMAE)
    and quadratic weighted kappa (QWK).
    """
    k = cm.shape[0]
    total = cm.sum()
    dist = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :])
    tp = np.diag(cm)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    rows = cm.sum(1)
    present = rows > 0
    mae_per_class = (cm * dist).sum(1)[present] / rows[present]
    w = dist ** 2 / (k - 1) ** 2
    expected = np.outer(rows, cm.sum(0)) / total
    recall = np.divide(tp, rows, out=np.full(k, np.nan), where=rows > 0)
    return {
        **{f"recall_{c + 1}": float(recall[c]) for c in range(k)},
        "accuracy": float(tp.sum() / total),
        "f1": float(f1.mean()),
        "mean_absolute_error": float((cm * dist).sum() / total),
        "one_off_accuracy": float(cm[dist <= 1].sum() / total),
        "amae": float(mae_per_class.mean()),
        "mmae": float(mae_per_class.max()),
        "qwk": float(1.0 - (w * cm).sum() / (w * expected).sum()),
    }


def discrete_metrics_vs_raters(pred: np.ndarray, rater_lo: np.ndarray,
                               rater_hi: np.ndarray) -> Dict[str, float]:
    """
    Discrete metrics of predictions (1..5) against the two raters, each rater weighted 0.5
    (soft confusion matrix, no rounding of the mean rating).
    """
    return _metrics_from_confusion(soft_confusion(pred, rater_lo, rater_hi))


def decode_thresholds(cum: np.ndarray, thresholds) -> np.ndarray:
    """Class 1..5 = 1 + number of P(Y > k) above its threshold (0.5 for all k = the PDF's rule)."""
    return 1 + (np.asarray(cum) > np.asarray(thresholds)[None, :]).sum(axis=1)


def tune_thresholds(cum: np.ndarray, rater_lo: np.ndarray, rater_hi: np.ndarray,
                    grid: Optional[np.ndarray] = None, rounds: int = 4) -> np.ndarray:
    """
    Per-threshold cut points t_k on P(Y > k) that maximise QWK against both raters,
    by coordinate ascent from 0.5 over a 0.01 grid (meant for validation predictions).
    """
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    cuts = np.full(np.asarray(cum).shape[1], 0.5)
    best = discrete_metrics_vs_raters(decode_thresholds(cum, cuts), rater_lo, rater_hi)["qwk"]
    for _ in range(rounds):
        changed = False
        for k in range(len(cuts)):
            for c in grid:
                trial = cuts.copy()
                trial[k] = c
                q = discrete_metrics_vs_raters(decode_thresholds(cum, trial), rater_lo, rater_hi)["qwk"]
                if q > best + 1e-9:
                    best, cuts, changed = q, trial, True
        if not changed:
            break
    return cuts


def human_reference_metrics(rater_lo: np.ndarray, rater_hi: np.ndarray) -> Dict[str, float]:
    """
    The same discrete metrics with one rater used as the "prediction" for the other
    (both directions, 0.5 each): what a single human scores against the other human.
    """
    cm = 0.5 * (soft_confusion(rater_hi, rater_lo, rater_lo) + soft_confusion(rater_lo, rater_hi, rater_hi))
    return _metrics_from_confusion(cm)


def krippendorff_alpha(units: np.ndarray, metric: str = "interval") -> float:
    """
    Krippendorff's alpha for units rated by exactly two coders (array [n_units, 2], no missing).
    metric: 'interval' (squared difference) or 'ordinal' (squared difference of mid-ranks).
    """
    x = np.asarray(units, dtype=np.float64)
    if metric == "ordinal":
        vals, counts = np.unique(x, return_counts=True)
        midrank = np.cumsum(counts) - counts / 2.0
        x = midrank[np.searchsorted(vals, x)]
    elif metric != "interval":
        raise ValueError(f"unknown metric {metric}")
    v = x.ravel()
    n = len(v)
    observed = np.mean((x[:, 0] - x[:, 1]) ** 2)
    expected = (2 * n * np.sum(v ** 2) - 2 * np.sum(v) ** 2) / (n * (n - 1))
    return float(1.0 - observed / expected)


def ordinal_metrics(cum: np.ndarray, rater_lo: np.ndarray, rater_hi: np.ndarray,
                    pred_override: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Metrics for one set of predictions given cumulative probabilities P(Y > k).

    Discrete class 1 + sum_k I[P(Y > k) > 0.5] (or pred_override, e.g. CE argmax), scored
    against both raters: accuracy, macro-F1, MAE, one-off, AMAE, MMAE, QWK, plus
    Krippendorff's alpha with the model treated as one more rater.
    Continuous score 1 + sum_k P(Y > k) vs the mean rating: MAE, RMSE, R2, Pearson,
    Spearman, Kendall tau-b.
    Probabilities vs the soft cumulative targets: mean BCE (comparable across heads),
    Brier score and AUC (unanimous points only) per threshold, and whether the
    predicted spread tracks rater disagreement.
    human_* keys: the same discrete metrics and alpha between the two raters, and the
    correlation ceiling sqrt(ICC(1,2)) of the mean rating, on the same points.
    """
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    cum = np.asarray(cum, dtype=np.float64)
    rater_lo = np.asarray(rater_lo, dtype=np.int64)
    rater_hi = np.asarray(rater_hi, dtype=np.int64)
    pred = np.asarray(pred_override if pred_override is not None else 1 + (cum > 0.5).sum(axis=1),
                      dtype=np.int64)
    score = 1.0 + cum.sum(axis=1)
    mean_rating = 0.5 * (rater_lo + rater_hi)
    thresholds = np.arange(1, cum.shape[1] + 1)
    target = 0.5 * (rater_lo[:, None] > thresholds) + 0.5 * (rater_hi[:, None] > thresholds)

    out = discrete_metrics_vs_raters(pred, rater_lo, rater_hi)
    model_vs_rater = np.concatenate([np.stack([pred, rater_lo], 1), np.stack([pred, rater_hi], 1)])
    out["alpha_ordinal"] = krippendorff_alpha(model_vs_rater, "ordinal")
    out["alpha_interval"] = krippendorff_alpha(model_vs_rater, "interval")

    out.update({
        "score_mae": float(np.mean(np.abs(score - mean_rating))),
        "score_rmse": float(np.sqrt(np.mean((score - mean_rating) ** 2))),
        "score_r2": float(1.0 - np.sum((score - mean_rating) ** 2) / np.sum((mean_rating - mean_rating.mean()) ** 2)),
        "score_pearson": float(pearsonr(score, mean_rating)[0]),
        "score_spearman": float(spearmanr(score, mean_rating).correlation),
        "score_kendall": float(kendalltau(score, mean_rating).correlation),
    })

    eps = 1e-7
    pc = np.clip(cum, eps, 1 - eps)
    out["cum_bce"] = float(-np.mean(target * np.log(pc) + (1 - target) * np.log(1 - pc)))
    for j, k in enumerate(thresholds):
        out[f"brier_t{k}"] = float(np.mean((cum[:, j] - target[:, j]) ** 2))
        unanimous = (target[:, j] == 0) | (target[:, j] == 1)
        labels = target[unanimous, j]
        out[f"auc_t{k}"] = (float(roc_auc_score(labels, cum[unanimous, j]))
                            if 0 < labels.sum() < len(labels) else float("nan"))
    probs = _class_probs(cum)
    classes = np.arange(1, probs.shape[1] + 1)
    pred_sd = np.sqrt((probs * (classes - (probs * classes).sum(1, keepdims=True)) ** 2).sum(1))
    out["sd_vs_disagreement_spearman"] = float(spearmanr(pred_sd, rater_hi - rater_lo).correlation)
    out["monotonicity_violation_pct"] = float(100 * np.mean((np.diff(cum, axis=1) > 0).any(axis=1)))

    human = human_reference_metrics(rater_lo, rater_hi)
    out.update({f"human_{k}": v for k, v in human.items()})
    pairs = np.stack([rater_lo, rater_hi], 1)
    out["human_alpha_ordinal"] = krippendorff_alpha(pairs, "ordinal")
    out["human_alpha_interval"] = krippendorff_alpha(pairs, "interval")
    n = len(pairs)
    msb = 2 * np.sum((pairs.mean(1) - pairs.mean()) ** 2) / (n - 1)
    msw = np.sum((pairs - pairs.mean(1, keepdims=True)) ** 2) / n
    out["human_corr_ceiling"] = float(np.sqrt(max(0.0, (msb - msw) / msb)))
    out["n"] = int(n)
    return out
