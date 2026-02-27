"""
verification/risk_scoring.py
-----------------------------
Enterprise risk scoring for verification pipeline.

Risk Score = weighted combination of:
  Drift (PSI)      50%
  KS Test          30%
  Bias Score       20%

Score range: 0–100
"""

from __future__ import annotations


def calculate_risk_score(
    psi: float,
    p_value: float,
    bias_score: float = 0.0,
) -> float:
    """
    Compute a 0–100 risk score from drift, statistical test, and bias metrics.

    Parameters
    ----------
    psi         Population Stability Index (0–∞, capped at 1.0 for scoring)
    p_value     KS test p-value (0–1); low p-value = more drift
    bias_score  Statistical parity difference or disparate impact deviation (0–1)

    Returns
    -------
    float  risk score in [0, 100]
    """
    # PSI: cap at 0.50 (anything above is max drift) then scale to 0–100
    drift_component = min(psi / 0.50, 1.0) * 100

    # KS: p-value near 0 = high drift; invert and scale
    ks_component = (1.0 - min(max(p_value, 0.0), 1.0)) * 100

    # Bias: scale absolute bias to 0–100
    bias_component = min(abs(bias_score) / 0.5, 1.0) * 100

    risk = (
        0.50 * drift_component
        + 0.30 * ks_component
        + 0.20 * bias_component
    )

    return round(max(0.0, min(risk, 100.0)), 2)
