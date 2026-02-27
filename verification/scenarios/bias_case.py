"""verification/scenarios/bias_case.py"""
import numpy as np
def generate_bias_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    protected = rng.integers(0, 2, n)
    # Group 1 gets higher positive rate — simulates biased model
    y_pred = np.where(
        protected == 1,
        rng.choice([0, 1], n, p=[0.25, 0.75]),
        rng.choice([0, 1], n, p=[0.45, 0.55]),
    )
    return y_pred, protected
