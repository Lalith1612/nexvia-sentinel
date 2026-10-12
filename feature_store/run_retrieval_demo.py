"""
Feast Feature Store End-to-End Retrieval Demonstration
Nexvia Sentinel AIML-01 - Phase 3

Demonstrates:
1. Historical point-in-time feature retrieval (offline store: Parquet).
2. Prevention of temporal leakage (queries before event timestamps return nulls).
3. Materialization to local online store (SQLite).
4. Real-time online feature retrieval for known and missing entities.
5. Schema comparison with SentinelFeaturePipeline.
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from feast import FeatureStore

REPO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_DIR.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from features import account_id, account_hourly_stats_view
from ml.features.feature_pipeline import FEATURE_NAMES as SENTINEL_FEATURES


def run_feast_demonstration():
    print("=" * 70)
    print("NEXVIA SENTINEL - FEAST FEATURE STORE DEMONSTRATION (PHASE 3)")
    print("=" * 70)

    # 1. Initialize FeatureStore
    store = FeatureStore(repo_path=str(REPO_DIR))
    print(f"[OK] FeatureStore initialized from {REPO_DIR}")
    print(f"     Project: {store.project}")
    print(f"     Registry: {store.config.registry}")

    # 2. Register entities & views
    store.apply([account_id, account_hourly_stats_view])
    print("[OK] Entities and FeatureViews applied to registry.")

    # 3. Load sample entities from source parquet
    parquet_path = REPO_DIR / "data" / "account_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Source parquet missing at {parquet_path}. Run generate_source_data.py first.")

    df_src = pd.read_parquet(parquet_path)
    sample_accounts = df_src["account_id"].head(2).tolist()
    sample_timestamps = df_src["event_timestamp"].head(2).tolist()
    print(f"\n[Test Accounts]: {sample_accounts}")
    print(f"[Source Event Timestamps]: {[t.isoformat() for t in sample_timestamps]}")

    # 4. Demonstrate Historical Retrieval (Point-in-Time Join)
    print("\n" + "-" * 50)
    print("STEP 1: Offline Historical Point-in-Time Feature Retrieval")
    print("-" * 50)

    # Query 1 hour AFTER event timestamp (features should be found)
    query_entity_df = pd.DataFrame({
        "account_id": sample_accounts,
        "event_timestamp": [t + timedelta(hours=1) for t in sample_timestamps]
    })

    feature_refs = [
        "account_hourly_stats:rolling_avg_amount_24h",
        "account_hourly_stats:rolling_txn_count_24h",
        "account_hourly_stats:rolling_max_amount_24h",
        "account_hourly_stats:zero_balance_frequency_24h",
        "account_hourly_stats:merchant_interaction_ratio_24h",
    ]

    hist_data = store.get_historical_features(
        entity_df=query_entity_df,
        features=feature_refs,
    ).to_df()

    print("Historical Feature Retrieval Results:")
    print(hist_data[["account_id", "event_timestamp", "rolling_avg_amount_24h", "rolling_txn_count_24h"]])

    # 5. Temporal Leakage Prevention Check
    print("\n" + "-" * 50)
    print("STEP 2: Temporal Leakage Prevention Verification")
    print("-" * 50)

    # Query 1 hour BEFORE the event timestamp (features should NOT be visible)
    leakage_test_df = pd.DataFrame({
        "account_id": [sample_accounts[0]],
        "event_timestamp": [sample_timestamps[0] - timedelta(hours=1)]
    })

    leakage_data = store.get_historical_features(
        entity_df=leakage_test_df,
        features=feature_refs,
    ).to_df()

    if len(leakage_data) == 0:
        print("Query at t - 1h: 0 records matched (Feast point-in-time join strictly dropped future data).")
    else:
        val = leakage_data["rolling_avg_amount_24h"].iloc[0]
        print(f"Query at t - 1h: rolling_avg_amount_24h = {val} (NaN indicates zero future leakage)")
        assert pd.isna(val), "ERROR: Historical point-in-time join leaked future information!"
    print("[PASS] Temporal leakage successfully prevented.")

    # 6. Materialization to Online Store
    print("\n" + "-" * 50)
    print("STEP 3: Materialization into Local Online Store (SQLite)")
    print("-" * 50)

    start_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    print(f"Materializing time window: {start_date} to {end_date}...")
    store.materialize(start_date=start_date, end_date=end_date)
    print("[PASS] Materialization to SQLite complete.")

    # 7. Online Feature Retrieval
    print("\n" + "-" * 50)
    print("STEP 4: Real-time Online Feature Retrieval")
    print("-" * 50)

    entity_rows = [
        {"account_id": sample_accounts[0]},
        {"account_id": sample_accounts[1]},
        {"account_id": "UNKNOWN_ACCOUNT_999"},  # Missing entity test
    ]

    online_features = store.get_online_features(
        features=feature_refs,
        entity_rows=entity_rows,
    ).to_dict()

    print("Online Features for entities:")
    for i, acc in enumerate(online_features["account_id"]):
        avg_amt = online_features["rolling_avg_amount_24h"][i]
        txn_cnt = online_features["rolling_txn_count_24h"][i]
        print(f"  Account {acc}: rolling_avg_amount_24h={avg_amt}, rolling_txn_count_24h={txn_cnt}")

    # 8. Schema Comparison
    print("\n" + "-" * 50)
    print("STEP 5: Schema Comparison with SentinelFeaturePipeline")
    print("-" * 50)
    feast_feature_names = [f.name for f in account_hourly_stats_view.schema]
    print(f"Feast Feature View ({len(feast_feature_names)} features):")
    for f in feast_feature_names:
        print(f"  - {f}")

    print(f"\nSentinelFeaturePipeline ({len(SENTINEL_FEATURES)} features):")
    for f in SENTINEL_FEATURES:
        print(f"  - {f}")

    print("\nArchitectural Isolation Summary:")
    print("  1. SentinelFeaturePipeline computes 20 transaction-instantaneous pre-auth signals.")
    print("  2. Feast account_hourly_stats manages multi-hour historical behavioral profiles.")
    print("  3. Production serving path remains safely decoupled, preventing any disruption.")
    print("=" * 70)


if __name__ == "__main__":
    run_feast_demonstration()
