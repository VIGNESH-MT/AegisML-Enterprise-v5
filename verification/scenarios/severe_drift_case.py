"""verification/scenarios/severe_drift_case.py"""
import numpy as np
def generate_severe_drift_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    train = rng.normal(0.50, 0.15, n).clip(0, 1)
    prod  = rng.normal(0.80, 0.12, n).clip(0, 1)  # large mean + std shift
    return train, prod
