"""
Automated Test Suite for Phase 1: Data Acquisition, Quality, Leakage & Temporal Splitting
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

TRAIN_PATH = PROCESSED_DIR / "train_benchmark_sample.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
METADATA_PATH = REPORTS_DIR / "split_metadata.json"
QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.json"

EXPECTED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"
]

@pytest.fixture(scope="module")
def train_df():
    assert TRAIN_PATH.exists(), f"Missing {TRAIN_PATH}"
    return pd.read_parquet(TRAIN_PATH)

@pytest.fixture(scope="module")
def val_df():
    assert VAL_PATH.exists(), f"Missing {VAL_PATH}"
    return pd.read_parquet(VAL_PATH)

@pytest.fixture(scope="module")
def test_df():
    assert TEST_PATH.exists(), f"Missing {TEST_PATH}"
    return pd.read_parquet(TEST_PATH)

@pytest.fixture(scope="module")
def split_metadata():
    assert METADATA_PATH.exists(), f"Missing {METADATA_PATH}"
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def test_schema_and_column_integrity(train_df, val_df, test_df):
    """Verify that all partitions adhere to the defined schema and column order."""
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert list(df.columns) == EXPECTED_COLUMNS, f"Schema mismatch in {name} partition"
        assert df["amount"].dtype == "float32", f"amount is not float32 in {name}"
        assert df["step"].dtype == "int16", f"step is not int16 in {name}"
        assert df["isFraud"].dtype == "int8", f"isFraud is not int8 in {name}"


def test_leakage_exclusion(train_df, val_df, test_df):
    """Verify that isFlaggedFraud is strictly excluded to prevent rule leakage."""
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert "isFlaggedFraud" not in df.columns, f"Rule leakage: isFlaggedFraud found in {name} partition"


def test_no_null_values(train_df, val_df, test_df):
    """Verify zero missing values across all columns and partitions."""
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        null_counts = df.isnull().sum()
        assert (null_counts == 0).all(), f"Found null values in {name} partition: {null_counts[null_counts > 0]}"


def test_target_integrity(train_df, val_df, test_df):
    """Verify ground truth target binary distribution and positive instance presence."""
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        unique_targets = set(df["isFraud"].unique())
        assert unique_targets.issubset({0, 1}), f"Non-binary target in {name}: {unique_targets}"
        assert (df["isFraud"] == 1).sum() > 0, f"No fraud instances found in {name}"
        assert (df["isFraud"] == 0).sum() > 0, f"No legitimate instances found in {name}"


def test_chronological_split_boundaries(train_df, val_df, test_df):
    """
    Verify strict chronological separation without overlap:
    Train <= 500 < Val (501..620) < Test (621..743)
    """
    max_train_step = int(train_df["step"].max())
    min_val_step = int(val_df["step"].min())
    max_val_step = int(val_df["step"].max())
    min_test_step = int(test_df["step"].min())
    max_test_step = int(test_df["step"].max())

    assert max_train_step <= 500, f"Train step leakage: max step is {max_train_step} (> 500)"
    assert min_val_step >= 501, f"Val step leakage: min step is {min_val_step} (< 501)"
    assert max_val_step <= 620, f"Val step leakage: max step is {max_val_step} (> 620)"
    assert min_test_step >= 621, f"Test step leakage: min step is {min_test_step} (< 621)"
    assert max_test_step <= 743, f"Test step invalid: max step is {max_test_step} (> 743)"

    # Direct disjoint check
    train_steps = set(train_df["step"].unique())
    val_steps = set(val_df["step"].unique())
    test_steps = set(test_df["step"].unique())

    assert train_steps.isdisjoint(val_steps), "Temporal overlap between Train and Val"
    assert val_steps.isdisjoint(test_steps), "Temporal overlap between Val and Test"
    assert train_steps.isdisjoint(test_steps), "Temporal overlap between Train and Test"


def test_test_set_natural_distribution(test_df, split_metadata):
    """Verify test set preserves un-sampled natural distribution (90,829 rows, 1,366 frauds)."""
    assert len(test_df) == 90_829, f"Unexpected test set count: {len(test_df)}"
    assert int((test_df["isFraud"] == 1).sum()) == 1_366, "Unexpected test set fraud count"
    fraud_rate = (test_df["isFraud"] == 1).sum() / len(test_df) * 100
    assert 1.45 <= fraud_rate <= 1.55, f"Test fraud rate deviated: {fraud_rate:.4f}%"


def test_fraud_preservation_in_benchmark_train(train_df, split_metadata):
    """Verify 100% of all 5,561 training frauds were preserved in the benchmark sample."""
    train_frauds = int((train_df["isFraud"] == 1).sum())
    expected_frauds = split_metadata["partitions"]["train_full_universe"]["fraud_count"]
    assert train_frauds == expected_frauds == 5_561, (
        f"Fraud loss in benchmark train: expected {expected_frauds}, got {train_frauds}"
    )


def test_metadata_audit_consistency(split_metadata):
    """Verify metadata report integrity and leakage audit flags."""
    audit = split_metadata["leakage_auditing"]
    assert audit["isFraud_included_in_features"] is False
    assert audit["isFlaggedFraud_included_in_features"] is False
    assert audit["temporal_overlap"] is False
    assert audit["max_train_step"] < audit["min_val_step"]
    assert audit["max_val_step"] < audit["min_test_step"]
