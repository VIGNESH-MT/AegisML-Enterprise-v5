"""
verification/metrics.py
------------------------
Drift detection metrics: PSI and KS Test.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI).

    PSI < 0.10   → No significant shift
    PSI 0.10–0.25 → Moderate shift — monitor
    PSI > 0.25   → Significant shift — investigate

    Returns
    -------
    float  PSI value (non-negative)
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)

    # Build bins on expected distribution
    _, bin_edges = np.histogram(expected, bins=bins)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts,   _ = np.histogram(actual,   bins=bin_edges)

    # Convert to proportions, add epsilon to prevent log(0)
    eps = 1e-8
    expected_pct = expected_counts / (len(expected) + eps) + eps
    actual_pct   = actual_counts   / (len(actual)   + eps) + eps

    psi = float(np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct)))
    return max(0.0, round(psi, 6))


def calculate_ks(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov two-sample test.

    Returns
    -------
    (ks_statistic, p_value)
    p-value < 0.05 → distributions are significantly different
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)
    statistic, p_value = ks_2samp(expected, actual)
    return round(float(statistic), 6), round(float(p_value), 6)


def interpret_psi(psi: float) -> str:
    if psi < 0.10:
        return "No significant shift — distribution is stable."
    elif psi < 0.25:
        return "Moderate shift — monitor closely and consider retraining schedule."
    else:
        return "Significant shift — immediate investigation and retraining required."


def interpret_ks(p_value: float) -> str:
    if p_value >= 0.05:
        return "Distributions are statistically similar (p >= 0.05)."
    elif p_value >= 0.01:
        return "Moderate distributional difference detected (p < 0.05)."
    else:
        return "Strong distributional difference detected (p < 0.01) — action required."
