"""
Tests for Phase 3 Feast Feature Store Implementation.

Verifies:
1. Valid repository configuration (feature_store.yaml).
2. Entity and FeatureView schema validity and absence of target/settlement leakage.
3. Offline point-in-time historical feature retrieval.
4. Temporal leakage prevention (past entity queries do not receive future data).
5. Online store materialization and real-time retrieval.
6. Handling of missing/unseen entities.
7. Schema isolation and consistency relative to SentinelFeaturePipeline.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import yaml
import pytest
import pandas as pd
from feast import FeatureStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEAST_REPO_DIR = REPO_ROOT / "feature_store"


@pytest.fixture(scope="module")
def feast_store():
    """Initializes and returns the Feast FeatureStore instance."""
    store = FeatureStore(repo_path=str(FEAST_REPO_DIR))
    from feature_store.features import account_id, account_hourly_stats_view
    store.apply([account_id, account_hourly_stats_view])
    return store


@pytest.fixture(scope="module")
def source_sample():
    """Reads a small sample from the generated feature source parquet."""
    parquet_path = FEAST_REPO_DIR / "data" / "account_features.parquet"
    assert parquet_path.exists(), "Source parquet must exist before running tests."
    df = pd.read_parquet(parquet_path)
    return df


def test_feature_store_yaml_validity():
    """Verifies feature_store.yaml configuration matches local requirements."""
    yaml_path = FEAST_REPO_DIR / "feature_store.yaml"
    assert yaml_path.exists(), "feature_store.yaml is missing."

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config.get("project") == "sentinel_feature_store"
    assert config.get("provider") == "local"
    assert config.get("online_store", {}).get("type") == "sqlite"
    assert config.get("offline_store", {}).get("type") == "file"


def test_feature_view_definitions(feast_store):
    """Verifies entity definitions, schema types, and leakage-safe fields."""
    fvs = {fv.name: fv for fv in feast_store.list_feature_views()}
    assert "account_hourly_stats" in fvs, "FeatureView 'account_hourly_stats' not registered."

    view = fvs["account_hourly_stats"]
    feature_names = [f.name for f in view.features]

    expected_features = {
        "rolling_avg_amount_24h",
        "rolling_txn_count_24h",
        "rolling_max_amount_24h",
        "zero_balance_frequency_24h",
        "merchant_interaction_ratio_24h",
    }
    assert set(feature_names) == expected_features

    # Verify zero post-settlement leakage fields
    leakage_forbidden = {"isFraud", "isFlaggedFraud", "newbalanceOrig", "newbalanceDest"}
    assert not (set(feature_names) & leakage_forbidden), "Leakage forbidden fields found in FeatureView!"


def test_offline_historical_retrieval(feast_store, source_sample):
    """Verifies point-in-time offline feature retrieval returns correct types and rows."""
    row = source_sample.iloc[0]
    acc_id = row["account_id"]
    ts = row["event_timestamp"]

    # Query 1 hour after the event
    entity_df = pd.DataFrame({
        "account_id": [acc_id],
        "event_timestamp": [ts + timedelta(hours=1)]
    })

    features = [
        "account_hourly_stats:rolling_avg_amount_24h",
        "account_hourly_stats:rolling_txn_count_24h",
        "account_hourly_stats:rolling_max_amount_24h",
        "account_hourly_stats:zero_balance_frequency_24h",
        "account_hourly_stats:merchant_interaction_ratio_24h",
    ]

    retrieved = feast_store.get_historical_features(
        entity_df=entity_df,
        features=features
    ).to_df()

    assert len(retrieved) == 1
    assert retrieved["account_id"].iloc[0] == acc_id
    assert not pd.isna(retrieved["rolling_avg_amount_24h"].iloc[0])
    assert retrieved["rolling_txn_count_24h"].iloc[0] >= 1


def test_temporal_leakage_prevention(feast_store, source_sample):
    """Verifies queries strictly prior to an entity event timestamp receive NaN (no future lookahead)."""
    row = source_sample.iloc[0]
    acc_id = row["account_id"]
    ts = row["event_timestamp"]

    # Query 1 hour BEFORE the event
    entity_df = pd.DataFrame({
        "account_id": [acc_id],
        "event_timestamp": [ts - timedelta(hours=1)]
    })

    features = ["account_hourly_stats:rolling_avg_amount_24h"]
    retrieved = feast_store.get_historical_features(
        entity_df=entity_df,
        features=features
    ).to_df()

    if len(retrieved) == 0:
        # Feast file offline point-in-time join strictly dropped the record because event_timestamp is before data
        assert True
    else:
        val = retrieved["rolling_avg_amount_24h"].iloc[0]
        assert pd.isna(val), f"Expected NaN for prior timestamp to prevent temporal leakage, got {val}"


def test_online_materialization_and_retrieval(feast_store, source_sample):
    """Verifies SQLite materialization and real-time online feature lookup."""
    start_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    feast_store.materialize(start_date=start_date, end_date=end_date)

    row = source_sample.iloc[0]
    acc_id = row["account_id"]

    features = [
        "account_hourly_stats:rolling_avg_amount_24h",
        "account_hourly_stats:rolling_txn_count_24h",
    ]

    online = feast_store.get_online_features(
        features=features,
        entity_rows=[{"account_id": acc_id}]
    ).to_dict()

    assert online["account_id"] == [acc_id]
    assert online["rolling_avg_amount_24h"][0] is not None
    assert online["rolling_txn_count_24h"][0] is not None
    assert online["rolling_txn_count_24h"][0] >= 1


def test_missing_entity_handling(feast_store):
    """Verifies online lookup for unseen entities safely returns None without raising exceptions."""
    features = [
        "account_hourly_stats:rolling_avg_amount_24h",
        "account_hourly_stats:rolling_txn_count_24h",
    ]

    online = feast_store.get_online_features(
        features=features,
        entity_rows=[{"account_id": "NON_EXISTENT_ACC_TEST_000"}]
    ).to_dict()

    assert online["account_id"] == ["NON_EXISTENT_ACC_TEST_000"]
    assert online["rolling_avg_amount_24h"][0] is None
    assert online["rolling_txn_count_24h"][0] is None


def test_schema_isolation_relative_to_sentinel_pipeline(feast_store):
    """Verifies Feast features do not mutate or collide with SentinelFeaturePipeline 20 features."""
    from ml.features.feature_pipeline import FEATURE_NAMES as SENTINEL_FEATURES

    view = feast_store.get_feature_view("account_hourly_stats")
    feast_features = set(f.name for f in view.features)

    # Confirm the 20 core SentinelFeaturePipeline features are preserved
    assert len(SENTINEL_FEATURES) == 20
    assert "amount" in SENTINEL_FEATURES
    assert "orig_balance_depletion" in SENTINEL_FEATURES
    assert "hour_of_day" in SENTINEL_FEATURES

    # Confirm Feast features are distinct rolling aggregations
    overlap = feast_features & set(SENTINEL_FEATURES)
    assert len(overlap) == 0, f"Unexpected feature collision between Feast and SentinelPipeline: {overlap}"
