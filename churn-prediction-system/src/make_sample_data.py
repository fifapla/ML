"""
Generates a small synthetic dataset matching the schema data_preprocessing.py
expects, so train.py is runnable out of the box. Replace data/dataset.csv
with a real churn dataset (e.g. the Telco Customer Churn dataset) for
anything beyond a pipeline smoke test - column names and churn-rate
realism here are illustrative, not based on real customer behavior.
"""
import os
import numpy as np
import pandas as pd

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


def make_sample_data(n=500, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], size=n),
        "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.5, 0.3, 0.2]),
        "payment_method": rng.choice(["Credit card", "Bank transfer", "Electronic check"], size=n),
        "tenure": rng.randint(0, 72, size=n),
        "monthly_charges": np.round(rng.uniform(20, 120, size=n), 2),
    })
    df["total_charges"] = np.round(df["tenure"] * df["monthly_charges"] * rng.uniform(0.9, 1.0, size=n), 2)

    # Churn is more likely for month-to-month, short-tenure, high-charge customers -
    # a simple synthetic signal so the model has something real to learn.
    churn_score = (
        (df["contract_type"] == "Month-to-month").astype(int) * 0.4
        + (df["tenure"] < 12).astype(int) * 0.3
        + (df["monthly_charges"] > 80).astype(int) * 0.2
        + rng.rand(n) * 0.3
    )
    df["churn"] = (churn_score > churn_score.median()).astype(int)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {n} synthetic rows to {OUT_PATH}")


if __name__ == "__main__":
    make_sample_data()
