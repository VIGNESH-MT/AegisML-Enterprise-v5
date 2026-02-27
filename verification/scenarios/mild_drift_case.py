"""verification/scenarios/mild_drift_case.py"""
import numpy as np
def generate_mild_drift_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    train = rng.normal(0.50, 0.15, n).clip(0, 1)
    prod  = rng.normal(0.58, 0.17, n).clip(0, 1)  # small mean shift
    return train, prod
