"""
Unit & Integration Tests for Sentinel Drift Monitoring Pipeline (Phase 1)
Verifies:
1. Normal drift: identical / non-drifted partitions evaluate cleanly without alert.
2. Detected drift: deliberate distribution shifts trigger drift flags and recommendation.
3. Missing dataset error handling.
4. Configurable drift threshold sensitivity.
5. Safety gate: automated retraining halts unless explicit operator approval is granted.
6. Report file generation: HTML and JSON artifacts are created on disk.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipelines.drift_monitor import SentinelDriftMonitor


@pytest.fixture
def sample_feature_df():
    """Generates synthetic baseline tabular feature DataFrame matching pipeline schema."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "amount": np.random.exponential(scale=1000.0, size=n).astype(np.float32),
        "log_amount": np.random.normal(loc=6.0, scale=1.5, size=n).astype(np.float32),
        "oldbalanceOrg": np.random.uniform(0, 50000, size=n).astype(np.float32),
        "log_oldbalanceOrg": np.random.normal(loc=8.0, scale=2.0, size=n).astype(np.float32),
        "oldbalanceDest": np.random.uniform(0, 100000, size=n).astype(np.float32),
        "log_oldbalanceDest": np.random.normal(loc=9.0, scale=2.0, size=n).astype(np.float32),
        "amount_to_oldbalance_ratio": np.random.uniform(0, 2.0, size=n).astype(np.float32),
        "orig_balance_depletion": np.random.choice([0.0, 1.0], size=n, p=[0.8, 0.2]).astype(np.float32),
        "orig_oldbalance_is_zero": np.random.choice([0.0, 1.0], size=n, p=[0.7, 0.3]).astype(np.float32),
        "dest_oldbalance_is_zero": np.random.choice([0.0, 1.0], size=n, p=[0.6, 0.4]).astype(np.float32),
        "is_dest_merchant": np.random.choice([0.0, 1.0], size=n, p=[0.5, 0.5]).astype(np.float32),
        "expected_newbalance_orig": np.random.uniform(0, 40000, size=n).astype(np.float32),
        "orig_deficit": np.random.uniform(0, 1000, size=n).astype(np.float32),
        "hour_of_day": np.random.uniform(0, 23, size=n).astype(np.float32),
        "day_of_week": np.random.uniform(0, 6, size=n).astype(np.float32),
        "type_CASH_IN": np.zeros(n, dtype=np.float32),
        "type_CASH_OUT": np.ones(n, dtype=np.float32),
        "type_DEBIT": np.zeros(n, dtype=np.float32),
        "type_PAYMENT": np.zeros(n, dtype=np.float32),
        "type_TRANSFER": np.zeros(n, dtype=np.float32),
    })


def test_normal_drift_within_threshold(sample_feature_df, tmp_path):
    """
    Verifies that comparing a dataset against a subset of itself yields zero or minimal
    statistical drift, resulting in dataset_drift_detected=False and no retraining recommendation.
    """
    monitor = SentinelDriftMonitor(drift_share_threshold=0.30, reports_dir=tmp_path)
    ref_df = sample_feature_df.iloc[:100].copy()
    cur_df = sample_feature_df.iloc[100:].copy()

    snapshot, summary = monitor.run_drift_analysis(ref_df, cur_df)

    assert summary["dataset_drift_detected"] is False
    assert summary["drift_share"] < 0.30
    assert summary["retraining_recommendation"]["action_required"] is False
    assert summary["retraining_recommendation"]["urgency"] == "NONE"


def test_detected_drift_triggers_retraining_recommendation(sample_feature_df, tmp_path):
    """
    Verifies that substantial distribution perturbation (shifting numerical features)
    is detected by the statistical drift test and triggers an ANALYST_REVIEW_AND_RETRAIN recommendation.
    """
    monitor = SentinelDriftMonitor(drift_share_threshold=0.25, reports_dir=tmp_path)
    ref_df = sample_feature_df.copy()

    # Create heavily drifted dataset by shifting multiple feature distributions
    drifted_df = sample_feature_df.copy()
    drifted_df["amount"] = drifted_df["amount"] * 100.0 + 50000.0
    drifted_df["log_amount"] = drifted_df["log_amount"] + 8.0
    drifted_df["oldbalanceOrg"] = drifted_df["oldbalanceOrg"] * 10.0
    drifted_df["log_oldbalanceOrg"] = drifted_df["log_oldbalanceOrg"] + 5.0
    drifted_df["oldbalanceDest"] = drifted_df["oldbalanceDest"] * 50.0
    drifted_df["log_oldbalanceDest"] = drifted_df["log_oldbalanceDest"] + 6.0
    drifted_df["hour_of_day"] = (drifted_df["hour_of_day"] + 12) % 24

    snapshot, summary = monitor.run_drift_analysis(ref_df, drifted_df)

    assert summary["dataset_drift_detected"] is True
    assert summary["drift_share"] >= 0.25
    assert summary["retraining_recommendation"]["action_required"] is True
    assert summary["retraining_recommendation"]["urgency"] in {"MEDIUM", "HIGH"}
    assert summary["retraining_recommendation"]["recommended_action"] == "ANALYST_REVIEW_AND_RETRAIN"
    assert "train_xgboost.py" in summary["retraining_recommendation"]["execution_command"]


def test_missing_dataset_files_raise_filenotfound(tmp_path):
    """Verifies that non-existent input dataset paths raise clean FileNotFoundError."""
    monitor = SentinelDriftMonitor(reports_dir=tmp_path)
    missing_ref = tmp_path / "non_existent_ref.parquet"
    missing_cur = tmp_path / "non_existent_cur.parquet"

    with pytest.raises(FileNotFoundError, match="Reference dataset not found"):
        monitor.load_datasets(missing_ref, missing_cur)


def test_configurable_threshold_sensitivity(sample_feature_df, tmp_path):
    """
    Verifies that drift sensitivity can be configured:
    A high threshold suppresses alerts, while a low threshold triggers alerts on mild shifts.
    """
    ref_df = sample_feature_df.copy()
    mild_drift_df = sample_feature_df.copy()
    # Mildly shift only 2 columns
    mild_drift_df["amount"] = mild_drift_df["amount"] * 2.0
    mild_drift_df["log_amount"] = mild_drift_df["log_amount"] + 1.0

    # With high threshold (0.90), drift should NOT be flagged
    strict_monitor = SentinelDriftMonitor(drift_share_threshold=0.90, reports_dir=tmp_path)
    _, summary_suppressed = strict_monitor.run_drift_analysis(ref_df, mild_drift_df)
    assert summary_suppressed["dataset_drift_detected"] is False

    # With very sensitive threshold (0.05), drift SHOULD be flagged
    sensitive_monitor = SentinelDriftMonitor(drift_share_threshold=0.05, reports_dir=tmp_path)
    _, summary_triggered = sensitive_monitor.run_drift_analysis(ref_df, mild_drift_df)
    assert summary_triggered["dataset_drift_detected"] is True


def test_safely_gated_retraining_prevents_unauthorized_execution(tmp_path):
    """
    CRITICAL SAFETY INVARIANT:
    Verifies that the retraining trigger is strictly gated.
    Even when drift is flagged, execution is blocked without explicit operator approval,
    and dry_run mode never mutates or overwrites model artifacts.
    """
    monitor = SentinelDriftMonitor(reports_dir=tmp_path)

    # 1. Action required but NO approval -> HALTED by safety gate
    rec_active = {
        "action_required": True,
        "urgency": "HIGH",
        "execution_command": "python ml/pipelines/train_xgboost.py --retrain"
    }
    gated_result = monitor.trigger_safely_gated_retraining(rec_active, approved=False)
    assert gated_result["triggered"] is False
    assert gated_result["status"] == "APPROVAL_REQUIRED"
    assert "Safety Gate Enforced" in gated_result["message"]

    # 2. Approved in dry_run mode -> SIMULATED without file mutation
    sim_result = monitor.trigger_safely_gated_retraining(rec_active, approved=True, dry_run=True)
    assert sim_result["triggered"] is True
    assert sim_result["status"] == "DRY_RUN_SIMULATION"
    assert "Production model artifacts in ml/models/ remained untouched" in sim_result["message"]

    # 3. No action required -> SUPPRESSED
    rec_inactive = {"action_required": False}
    no_act_result = monitor.trigger_safely_gated_retraining(rec_inactive, approved=True)
    assert no_act_result["triggered"] is False
    assert no_act_result["status"] == "NO_ACTION_REQUIRED"


def test_report_artifacts_generation(sample_feature_df, tmp_path):
    """Verifies that save_reports creates both HTML and JSON files with content."""
    monitor = SentinelDriftMonitor(reports_dir=tmp_path)
    ref_df = sample_feature_df.iloc[:100].copy()
    cur_df = sample_feature_df.iloc[100:].copy()

    snapshot, summary = monitor.run_drift_analysis(ref_df, cur_df)
    html_file, json_file = monitor.save_reports(snapshot, summary, html_name="test_report.html", json_name="test_summary.json")

    assert html_file.exists()
    assert html_file.stat().st_size > 1000
    assert json_file.exists()
    assert json_file.stat().st_size > 100

    import json
    with open(json_file, "r") as f:
        data = json.load(f)
    assert "dataset_drift_detected" in data
    assert "column_metrics" in data
    assert "retraining_recommendation" in data
