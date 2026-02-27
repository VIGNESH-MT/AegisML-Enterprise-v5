"""
verification/config.py
-----------------------
Centralised deterministic seed management and environment configuration.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_deterministic(seed: int = 42) -> None:
    """Set all random seeds for fully reproducible results."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Supported report languages
SUPPORTED_LANGUAGES = [
    "English", "German", "French", "Spanish", "Italian",
    "Polish", "Romanian", "Dutch", "Russian", "Ukrainian",
    "Hindi", "Urdu", "Arabic",
]

# Default output paths
VERIFICATION_REPORT_PATH = "verification/verification_report.json"
DEFAULT_AUDIT_PDF_PATH   = "verification/Audit_Report.pdf"

# Risk thresholds
PSI_STABLE      = 0.10
PSI_MODERATE    = 0.25
RISK_LOW_MAX    = 30
RISK_MEDIUM_MAX = 70
RISK_HIGH_MAX   = 90
