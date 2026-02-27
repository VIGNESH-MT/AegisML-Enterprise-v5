"""
create_test_data.py
--------------------
Generates realistic synthetic CSV datasets for testing AegisML.

Datasets created:
  - examples/stable_train.csv       Clean training distribution (credit scoring model)
  - examples/stable_prod.csv        Production data matching training distribution
  - examples/drifted_prod.csv       Production data with significant drift
  - examples/mild_drift_prod.csv    Production data with mild drift
  - examples/sample_predictions.csv Predictions with y_true and y_prob for Streamlit demo

Usage:
  python create_test_data.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEED = 42
N_TRAIN = 1000
N_PROD  = 500

os.makedirs("examples", exist_ok=True)


def generate_credit_scoring_features(n: int, seed: int, drift_factor: float = 0.0) -> pd.DataFrame:
    """
    Generate synthetic credit scoring feature data.

    Features:
      credit_score     (300–850)
      annual_income    (£20k–£250k)
      debt_to_income   (0–1)
      employment_years (0–30)
      loan_amount      (£5k–£100k)
      age              (18–75)
    """
    rng = np.random.default_rng(seed)

    credit_score     = rng.normal(680 + drift_factor * 50, 80, n).clip(300, 850)
    annual_income    = rng.lognormal(10.8 + drift_factor * 0.3, 0.5, n).clip(20_000, 250_000)
    debt_to_income   = rng.beta(2 + drift_factor, 6, n).clip(0.01, 0.99)
    employment_years = rng.exponential(5 + drift_factor, n).clip(0, 30)
    loan_amount      = rng.lognormal(10.5, 0.6, n).clip(5_000, 100_000)
    age              = rng.normal(42 + drift_factor * 5, 12, n).clip(18, 75)

    return pd.DataFrame({
        "credit_score":     credit_score.round(1),
        "annual_income":    annual_income.round(2),
        "debt_to_income":   debt_to_income.round(4),
        "employment_years": employment_years.round(1),
        "loan_amount":      loan_amount.round(2),
        "age":              age.round(1),
    })


def generate_predictions(n: int, seed: int, accuracy: float = 0.85) -> pd.DataFrame:
    """
    Generate synthetic model predictions (y_true, y_prob).

    Parameters
    ----------
    n        Number of samples
    seed     Random seed
    accuracy Target accuracy level
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)

    # Generate probabilities with controlled accuracy
    y_prob = np.where(
        y_true == 1,
        rng.beta(8 * accuracy, 2, n),
        rng.beta(2, 8 * accuracy, n),
    ).clip(0.001, 0.999)

    return pd.DataFrame({"y_true": y_true, "y_prob": y_prob.round(4)})


def main():
    print("Generating AegisML test datasets...\n")

    # 1. Stable training data (credit scoring features)
    train_df = generate_credit_scoring_features(N_TRAIN, seed=SEED, drift_factor=0.0)
    train_df.to_csv("examples/stable_train.csv", index=False)
    print(f"Created: examples/stable_train.csv  ({len(train_df)} rows)")

    # 2. Stable production data (same distribution)
    stable_prod_df = generate_credit_scoring_features(N_PROD, seed=SEED + 1, drift_factor=0.0)
    stable_prod_df.to_csv("examples/stable_prod.csv", index=False)
    print(f"Created: examples/stable_prod.csv   ({len(stable_prod_df)} rows)")

    # 3. Mild drift production data
    mild_prod_df = generate_credit_scoring_features(N_PROD, seed=SEED + 2, drift_factor=0.5)
    mild_prod_df.to_csv("examples/mild_drift_prod.csv", index=False)
    print(f"Created: examples/mild_drift_prod.csv ({len(mild_prod_df)} rows)")

    # 4. Significant drift production data (simulates economic shock)
    drifted_prod_df = generate_credit_scoring_features(N_PROD, seed=SEED + 3, drift_factor=2.0)
    drifted_prod_df.to_csv("examples/drifted_prod.csv", index=False)
    print(f"Created: examples/drifted_prod.csv  ({len(drifted_prod_df)} rows)")

    # 5. Sample predictions for Streamlit demo
    good_preds = generate_predictions(500, seed=SEED, accuracy=0.88)
    good_preds.to_csv("examples/sample_predictions.csv", index=False)
    print(f"Created: examples/sample_predictions.csv (500 rows, ~88% accuracy)")

    # 6. Degraded predictions
    degraded_preds = generate_predictions(500, seed=SEED + 10, accuracy=0.62)
    degraded_preds.to_csv("examples/degraded_predictions.csv", index=False)
    print(f"Created: examples/degraded_predictions.csv (500 rows, ~62% accuracy)")

    # 7. Reference predictions (for drift comparison)
    ref_preds = generate_predictions(500, seed=SEED + 20, accuracy=0.90)
    ref_preds[["y_prob"]].rename(columns={"y_prob": "y_prob"}).to_csv(
        "examples/reference_predictions.csv", index=False
    )
    print(f"Created: examples/reference_predictions.csv (500 rows reference)")

    print("\nAll test datasets generated successfully.")
    print("\nUsage examples:")
    print("  Stable audit:   train=stable_train.csv, prod=stable_prod.csv, column=credit_score")
    print("  Drift audit:    train=stable_train.csv, prod=drifted_prod.csv, column=credit_score")
    print("  Streamlit demo: predictions=sample_predictions.csv")


if __name__ == "__main__":
    main()
