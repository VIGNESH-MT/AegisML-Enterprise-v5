"""
reliability/__init__.py
------------------------
AegisML Enterprise — Reliability Module
"""

from .metrics import compute_metrics, format_confusion_matrix, compute_threshold_analysis
from .calibration import compute_ece, reliability_diagram, confidence_histogram, calibration_error_heatmap
from .drift import compute_prediction_drift, compute_feature_drift, drift_heatmap
from .risk import compute_risk_score
from .report import generate_report

__all__ = [
    # Metrics
    "compute_metrics",
    "format_confusion_matrix",
    "compute_threshold_analysis",
    # Calibration
    "compute_ece",
    "reliability_diagram",
    "confidence_histogram",
    "calibration_error_heatmap",
    # Drift
    "compute_prediction_drift",
    "compute_feature_drift",
    "drift_heatmap",
    # Risk
    "compute_risk_score",
    # Report
    "generate_report",
]
