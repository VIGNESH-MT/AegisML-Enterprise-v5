"""
reliability/risk.py
--------------------
Enterprise risk scoring engine with EU AI Act compliance mapping.

Risk Score = weighted combination of:
  - Calibration (ECE)          35%
  - Probabilistic accuracy (Brier Score)  25%
  - Predictive accuracy        20%
  - Distribution drift (PSI)   20%

Risk Levels: Low · Medium · High · Critical
EU AI Act Risk Categories: Minimal · Limited · High · Unacceptable
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds (tunable via env/config in future)
# ---------------------------------------------------------------------------

_T = {
    # ECE thresholds
    "ece_low":    0.05,
    "ece_medium": 0.10,
    "ece_high":   0.15,
    # Brier Score thresholds
    "brier_low":    0.10,
    "brier_medium": 0.17,
    "brier_high":   0.22,
    # Accuracy thresholds (inverted — higher is better)
    "acc_low":    0.90,
    "acc_medium": 0.80,
    "acc_high":   0.70,
    # Drift (PSI) thresholds
    "drift_low":    0.05,
    "drift_medium": 0.10,
    "drift_high":   0.20,
    # AUC thresholds (inverted)
    "auc_low":    0.90,
    "auc_medium": 0.80,
    "auc_high":   0.70,
}

RISK_LEVELS = ["Low", "Medium", "High", "Critical"]

# EU AI Act risk category mapping
_EU_AI_ACT_CATEGORIES = {
    "Low":      "Minimal Risk",
    "Medium":   "Limited Risk",
    "High":     "High Risk (Article 6/7 — requires conformity assessment)",
    "Critical": "Unacceptable Risk (Article 5 — deployment prohibited)",
}

# UK AI Framework mapping
_UK_AI_FRAMEWORK = {
    "Low":      "Standard Monitoring Sufficient",
    "Medium":   "Enhanced Monitoring + ICO Notification Advised",
    "High":     "Human Oversight Required — FCA/ICO Notification",
    "Critical": "Immediate Suspension — Regulatory Notification Required",
}


def _score_component(value: float, lo: float, med: float, hi: float, higher_is_worse: bool = True) -> int:
    """Map a metric value to a 0–3 integer risk score."""
    if higher_is_worse:
        if value <= lo:  return 0
        if value <= med: return 1
        if value <= hi:  return 2
        return 3
    else:
        if value >= lo:  return 0
        if value >= med: return 1
        if value >= hi:  return 2
        return 3


def compute_risk_score(
    ece: float,
    brier_score: float,
    accuracy: float,
    drift_score: float | None = None,
    auc_roc: float | None = None,
) -> dict:
    """
    Compute overall deployment risk score and category.

    Parameters
    ----------
    ece           Expected Calibration Error
    brier_score   Brier Score
    accuracy      Classification accuracy
    drift_score   PSI drift score (optional)
    auc_roc       ROC-AUC (optional, used as additional signal)

    Returns
    -------
    dict with risk level, scores, recommendations, verdicts
    """
    component_scores: dict[str, int] = {}

    component_scores["calibration_ece"] = _score_component(
        ece, _T["ece_low"], _T["ece_medium"], _T["ece_high"]
    )
    component_scores["brier_score"] = _score_component(
        brier_score, _T["brier_low"], _T["brier_medium"], _T["brier_high"]
    )
    component_scores["accuracy"] = _score_component(
        accuracy, _T["acc_low"], _T["acc_medium"], _T["acc_high"], higher_is_worse=False
    )

    if drift_score is not None:
        component_scores["distribution_drift"] = _score_component(
            drift_score, _T["drift_low"], _T["drift_medium"], _T["drift_high"]
        )

    if auc_roc is not None:
        component_scores["auc_roc"] = _score_component(
            auc_roc, _T["auc_low"], _T["auc_medium"], _T["auc_high"], higher_is_worse=False
        )

    # Weights
    base_weights = {
        "calibration_ece": 0.35,
        "brier_score":     0.25,
        "accuracy":        0.20,
        "distribution_drift": 0.15,
        "auc_roc":         0.05,
    }

    active_keys = [k for k in component_scores]
    total_w = sum(base_weights.get(k, 0.05) for k in active_keys)
    weighted_sum = sum(
        component_scores[k] * base_weights.get(k, 0.05) for k in active_keys
    )
    weighted_avg = weighted_sum / total_w if total_w > 0 else 0.0
    overall_idx = min(int(np.ceil(weighted_avg)), 3)
    risk_level = RISK_LEVELS[overall_idx]

    # Build component output
    component_output = {
        k: {"score": v, "level": RISK_LEVELS[v]}
        for k, v in component_scores.items()
    }

    recommendations = _build_recommendations(
        component_scores, ece, brier_score, accuracy, drift_score, auc_roc
    )
    deployment_verdict = _deployment_verdict(risk_level)

    return {
        "overall_risk_level": risk_level,
        "overall_score": overall_idx,
        "weighted_score": round(weighted_avg, 3),
        "component_scores": component_output,
        "recommendations": recommendations,
        "deployment_verdict": deployment_verdict,
        "eu_ai_act_category": _EU_AI_ACT_CATEGORIES[risk_level],
        "uk_ai_framework": _UK_AI_FRAMEWORK[risk_level],
        "requires_human_oversight": risk_level in ("High", "Critical"),
        "requires_regulatory_notification": risk_level == "Critical",
    }


def _build_recommendations(
    scores: dict,
    ece: float,
    brier: float,
    accuracy: float,
    drift: float | None,
    auc: float | None,
) -> list[str]:
    recs = []

    # Calibration
    if scores.get("calibration_ece", 0) >= 2:
        recs.append(
            f"CRITICAL — Calibration severely degraded (ECE={ece:.4f}). "
            "Apply temperature scaling or Platt scaling before deployment. "
            "EU AI Act Article 15 requires robustness assurance."
        )
    elif scores.get("calibration_ece", 0) == 1:
        recs.append(
            f"WARNING — Moderate calibration error (ECE={ece:.4f}). "
            "Monitor confidence scores in production. Consider isotonic regression recalibration."
        )

    # Brier Score
    if scores.get("brier_score", 0) >= 2:
        recs.append(
            f"WARNING — Brier Score ({brier:.4f}) indicates poor probabilistic predictions. "
            "Retrain with calibrated loss functions (e.g. cross-entropy with label smoothing)."
        )

    # Accuracy
    if scores.get("accuracy", 0) >= 2:
        recs.append(
            f"WARNING — Accuracy ({accuracy:.2%}) below acceptable threshold. "
            "Review training data quality, class balance, and feature engineering."
        )

    # Drift
    if drift is not None:
        d_score = scores.get("distribution_drift", 0)
        if d_score >= 2:
            recs.append(
                f"CRITICAL — Significant distribution drift detected (PSI={drift:.4f}). "
                "Initiate model retraining. Freeze deployment until investigation complete."
            )
        elif d_score == 1:
            recs.append(
                f"WARNING — Moderate distribution drift (PSI={drift:.4f}). "
                "Increase monitoring frequency. Schedule retraining assessment."
            )

    # AUC
    if auc is not None and scores.get("auc_roc", 0) >= 2:
        recs.append(
            f"WARNING — AUC-ROC ({auc:.4f}) indicates poor discrimination. "
            "Review model architecture and feature selection."
        )

    # All clear
    if not recs:
        recs.append(
            "OK — Model meets all reliability thresholds. Deployment approved. "
            "Maintain standard monitoring schedule and quarterly review."
        )
        recs.append(
            "RECOMMENDATION — Log all production predictions, configure alerting "
            "dashboards, and set up automated drift detection pipelines."
        )

    return recs


def _deployment_verdict(risk_level: str) -> str:
    verdicts = {
        "Low": (
            "Model demonstrates strong reliability across all assessed dimensions. "
            "Deployment is approved with standard monitoring. "
            "Schedule quarterly reliability review."
        ),
        "Medium": (
            "Model shows moderate reliability concerns in one or more dimensions. "
            "Conditional deployment approved with enhanced monitoring, "
            "confidence thresholding, and automated drift alerts in place."
        ),
        "High": (
            "Model exhibits significant reliability concerns. "
            "Deployment NOT recommended without remediation. "
            "Human oversight required per EU AI Act Article 14. "
            "Notify responsible AI team and governance committee."
        ),
        "Critical": (
            "CRITICAL RISK — Model must NOT be deployed. "
            "Multiple reliability dimensions are severely degraded. "
            "EU AI Act Article 9 risk management obligations triggered. "
            "Immediate escalation to governance committee required. "
            "Regulatory notification may be required under applicable law."
        ),
    }
    return verdicts.get(risk_level, "Unknown risk level.")
