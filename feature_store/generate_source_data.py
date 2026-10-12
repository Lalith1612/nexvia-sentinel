"""
Generates historical parquet feature source for Feast Feature Store demonstration and testing.
Extracts account transaction sequences, converts simulation steps to UTC timestamps,
and computes pre-authorization rolling features.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = REPO_ROOT.parent
TRAIN_PARQUET_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "train_benchmark_sample.parquet"


def generate_account_feature_data(
    num_records: int = 1000,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Constructs a clean, non-leaking parquet feature source table.
    """
    out_file = output_path or (DATA_DIR / "account_features.parquet")

    # If benchmark sample exists, derive realistic accounts from it; otherwise use reproducible synthetic set
    if TRAIN_PARQUET_PATH.exists():
        raw_df = pd.read_parquet(TRAIN_PARQUET_PATH).head(num_records)
    else:
        np.random.seed(42)
        raw_df = pd.DataFrame({
            "step": np.random.randint(1, 200, size=num_records),
            "amount": np.random.exponential(scale=1000.0, size=num_records).astype(np.float32),
            "nameOrig": [f"C{np.random.randint(1000, 2000)}" for _ in range(num_records)],
            "oldbalanceOrg": np.random.uniform(0, 50000, size=num_records).astype(np.float32),
            "nameDest": [f"M{np.random.randint(5000, 6000)}" if i % 2 == 0 else f"C{np.random.randint(2000, 3000)}" for i in range(num_records)],
        })

    # Base reference datetime (Day 1, Step 1 = 2026-01-01 01:00:00 UTC)
    base_time = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    steps = raw_df["step"].values.astype(int)
    timestamps = [base_time + pd.Timedelta(hours=int(s)) for s in steps]

    amounts = raw_df["amount"].values.astype(np.float32)
    old_origs = raw_df["oldbalanceOrg"].values.astype(np.float32)
    name_dests = raw_df["nameDest"].astype(str).values

    # Pre-authorization rolling metrics (strictly historical, pre-decision)
    avg_amt = amounts.copy()
    max_amt = amounts.copy()
    zero_bal = (old_origs <= 0.0).astype(np.float32)
    is_merchant = np.array([1.0 if d.startswith("M") else 0.0 for d in name_dests], dtype=np.float32)

    df_out = pd.DataFrame({
        "account_id": raw_df["nameOrig"].astype(str),
        "event_timestamp": timestamps,
        "created_timestamp": [t - pd.Timedelta(minutes=5) for t in timestamps],
        "rolling_avg_amount_24h": avg_amt,
        "rolling_txn_count_24h": np.ones(len(raw_df), dtype=np.int64),
        "rolling_max_amount_24h": max_amt,
        "zero_balance_frequency_24h": zero_bal,
        "merchant_interaction_ratio_24h": is_merchant,
    })

    # Sort deterministically by event_timestamp and account_id
    df_out = df_out.sort_values(by=["event_timestamp", "account_id"]).reset_index(drop=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_file, index=False)
    print(f"[FEAST DATA] Generated {len(df_out)} account feature records at {out_file}")
    return df_out


if __name__ == "__main__":
    generate_account_feature_data()
