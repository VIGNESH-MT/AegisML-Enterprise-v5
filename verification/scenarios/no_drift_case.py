"""verification/scenarios/no_drift_case.py"""
import numpy as np
def generate_no_drift_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    train = rng.normal(0.5, 0.15, n).clip(0, 1)
    prod  = rng.normal(0.5, 0.15, n).clip(0, 1)   # same distribution
    return train, prod
