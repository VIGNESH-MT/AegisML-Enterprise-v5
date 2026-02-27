"""verification/scenarios/failure_case.py"""
import numpy as np
def generate_small_sample_data(seed=42):
    rng = np.random.default_rng(seed)
    train = rng.normal(0.5, 0.1, 15)   # below 30-sample minimum
    prod  = rng.normal(0.6, 0.1, 15)
    return train, prod
