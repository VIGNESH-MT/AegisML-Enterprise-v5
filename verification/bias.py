"""
verification/bias.py
---------------------
Algorithmic fairness and bias assessment metrics.

Metrics implemented:
  - Statistical Parity Difference (SPD)
  - Disparate Impact Ratio (DIR)
  - Equal Opportunity Difference (EOD)
  - Average Odds Difference (AOD)
  - Predictive Parity (PP)

References:
  Feldman et al. (2015); Hardt et al. (2016); Chouldechova (2017)
  EU AI Act Annex III — non-discrimination requirements
  FCA Guidance: FG23/4 — Artificial Intelligence in Financial Services
"""

from __future__ import annotations

import numpy as np


def statistical_parity_difference(
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
) -> float:
    """
    Statistical Parity Difference (SPD).
    SPD = P(Ŷ=1 | A=1) − P(Ŷ=1 | A=0)

    Ideal: 0.0
    Acceptable range: |SPD| < 0.10 (EU AI Act / EEOC 80% rule)
    """
    y_pred = np.asarray(y_pred)
    protected_attr = np.asarray(protected_attr)

    group_1 = y_pred[protected_attr == 1]
    group_0 = y_pred[protected_attr == 0]

    if len(group_1) == 0 or len(group_0) == 0:
        return 0.0

    return round(float(np.mean(group_1) - np.mean(group_0)), 5)


def disparate_impact_ratio(
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
) -> float:
    """
    Disparate Impact Ratio (DIR).
    DIR = P(Ŷ=1 | A=1) / P(Ŷ=1 | A=0)

    Ideal: 1.0
    EEOC 4/5ths rule: DIR ≥ 0.80 considered acceptable.
    """
    y_pred = np.asarray(y_pred)
    protected_attr = np.asarray(protected_attr)

    group_1 = y_pred[protected_attr == 1]
    group_0 = y_pred[protected_attr == 0]

    if len(group_1) == 0 or len(group_0) == 0:
        return 1.0

    rate_0 = float(np.mean(group_0))
    if rate_0 == 0:
        return 1.0

    return round(float(np.mean(group_1)) / rate_0, 5)


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
) -> float:
    """
    Equal Opportunity Difference (EOD).
    EOD = TPR(A=1) − TPR(A=0)

    Ideal: 0.0  — equal true positive rates across groups.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected_attr = np.asarray(protected_attr)

    def _tpr(mask):
        positives = y_true[mask] == 1
        if positives.sum() == 0:
            return 0.0
        return float(y_pred[mask][positives].mean())

    tpr_1 = _tpr(protected_attr == 1)
    tpr_0 = _tpr(protected_attr == 0)
    return round(tpr_1 - tpr_0, 5)


def average_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
) -> float:
    """
    Average Odds Difference (AOD).
    AOD = 0.5 × [(FPR(A=1) − FPR(A=0)) + (TPR(A=1) − TPR(A=0))]

    Ideal: 0.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    protected_attr = np.asarray(protected_attr)

    def _rates(mask):
        yt, yp = y_true[mask], y_pred[mask]
        pos_mask = yt == 1
        neg_mask = yt == 0
        tpr = float(yp[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
        fpr = float(yp[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0
        return tpr, fpr

    tpr1, fpr1 = _rates(protected_attr == 1)
    tpr0, fpr0 = _rates(protected_attr == 0)

    return round(0.5 * ((fpr1 - fpr0) + (tpr1 - tpr0)), 5)


def compute_full_bias_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_attr: np.ndarray,
    group_names: tuple[str, str] = ("Group 0", "Group 1"),
) -> dict:
    """
    Full bias assessment report with all fairness metrics.

    Returns
    -------
    dict with all metrics, interpretations, compliance flags
    """
    spd = statistical_parity_difference(y_pred, protected_attr)
    dir_ratio = disparate_impact_ratio(y_pred, protected_attr)
    eod = equal_opportunity_difference(y_true, y_pred, protected_attr)
    aod = average_odds_difference(y_true, y_pred, protected_attr)

    # Compute bias score for risk engine (0–1 scale)
    bias_score = min(abs(spd) / 0.5, 1.0)

    # Compliance flags
    spd_compliant = abs(spd) < 0.10
    dir_compliant = dir_ratio >= 0.80
    eod_compliant = abs(eod) < 0.10

    overall_compliant = spd_compliant and dir_compliant and eod_compliant

    return {
        "statistical_parity_difference": spd,
        "disparate_impact_ratio": dir_ratio,
        "equal_opportunity_difference": eod,
        "average_odds_difference": aod,
        "bias_score": round(bias_score, 4),
        "group_names": group_names,
        "compliance": {
            "spd_compliant": spd_compliant,
            "dir_compliant_eeoc_4_5ths": dir_compliant,
            "eod_compliant": eod_compliant,
            "overall_compliant": overall_compliant,
        },
        "interpretations": {
            "spd": (
                f"SPD={spd:.4f}: "
                + ("Fair — groups receive similar positive rates." if abs(spd) < 0.10
                   else "UNFAIR — statistically significant disparity between groups.")
            ),
            "dir": (
                f"DIR={dir_ratio:.4f}: "
                + ("Acceptable — passes EEOC 4/5ths rule." if dir_ratio >= 0.80
                   else "FAILS EEOC 4/5ths rule — potential discriminatory impact.")
            ),
            "eod": (
                f"EOD={eod:.4f}: "
                + ("Equal opportunity maintained." if abs(eod) < 0.10
                   else "Unequal opportunity detected — investigate model decisions.")
            ),
        },
        "eu_ai_act_bias_obligation": (
            "COMPLIANT — No significant bias detected."
            if overall_compliant else
            "ACTION REQUIRED — EU AI Act Annex III non-discrimination requirements not met. "
            "Bias mitigation required before deployment."
        ),
        "fca_fairness_obligation": (
            "COMPLIANT" if overall_compliant else
            "REVIEW REQUIRED — FCA Consumer Duty fair outcomes obligation may be breached."
        ),
    }
