"""
reliability/metrics.py
-----------------------
Enterprise metric suite for binary classification models.

Metrics:
  - Accuracy, Precision, Recall, Specificity, F1 (weighted + macro)
  - ROC-AUC, Average Precision (PR-AUC)
  - Brier Score, Log Loss, Matthews Correlation Coefficient
  - Confusion matrix (raw + formatted)
  - Per-threshold analysis

All values are deterministic and rounded for stable reports.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like of int  (0/1)
    y_prob : array-like of float (probabilities in [0,1])
    threshold : float  Decision threshold (default 0.5)

    Returns
    -------
    dict with all metric values
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    # Core classification
    acc = float(accuracy_score(y_true, y_pred))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0

    # Probabilistic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logloss = float(log_loss(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 1.0

    # AUC metrics
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    if len(np.unique(y_true)) > 1:
        try:
            auc_roc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            auc_pr = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Risk interpretation
    def _risk_label(v, lo, hi):
        return "Low" if v <= lo else "High" if v >= hi else "Medium"

    return {
        # Classification
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "f1": round(f1_w, 4),            # backwards-compat alias
        "mcc": round(mcc, 4),
        "false_positive_rate": round(fpr, 4),
        "negative_predictive_value": round(npv, 4),
        # Probabilistic
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "auc_roc": round(auc_roc, 4) if auc_roc is not None else None,
        "auc_pr": round(auc_pr, 4) if auc_pr is not None else None,
        # Counts
        "n_samples": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "class_balance": round(n_pos / n, 4) if n > 0 else 0.0,
        # Threshold
        "threshold": threshold,
        # Confusion matrix
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        # Risk flags
        "flags": {
            "low_accuracy": acc < 0.70,
            "poor_brier": brier > 0.22,
            "poor_auc": (auc_roc is not None and auc_roc < 0.70),
            "high_logloss": logloss > 0.60,
            "class_imbalance": min(n_pos, n_neg) / n < 0.10 if n > 0 else False,
        },
    }


def format_confusion_matrix(cm: list | np.ndarray) -> str:
    """Return a human-readable confusion matrix string."""
    cm = np.array(cm)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return (
        "                  Predicted 0    Predicted 1\n"
        f"  Actual 0          {tn:<10}     {fp:<10}\n"
        f"  Actual 1          {fn:<10}     {tp:<10}"
    )


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[list] = None,
) -> list[dict]:
    """
    Analyse metric sensitivity across decision thresholds.
    Useful for recommending optimal threshold for deployment.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 1.0, 0.05)]

    results = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=t)
        results.append({
            "threshold": t,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_weighted": m["f1_weighted"],
            "specificity": m["specificity"],
            "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        })
    return results
