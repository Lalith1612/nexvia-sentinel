"""
Automated Test Suite for Phase 2 Audit Reproducibility and Invariant Verification
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys_path = str(PROJECT_ROOT)
import sys
if sys_path not in sys.path:
    sys.path.append(sys_path)

from ml.features.feature_pipeline import SentinelFeaturePipeline, FEATURE_NAMES

PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

TEST_PATH = PROCESSED_DIR / "test.parquet"
MODEL_PATH = MODELS_DIR / "xgboost_baseline.json"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"
CONFIG_PATH = MODELS_DIR / "model_config.json"
AUDIT_REPORT_PATH = REPORTS_DIR / "phase2_independent_audit.json"

@pytest.fixture(scope="module")
def audit_data():
    test_df = pd.read_parquet(TEST_PATH)
    pipeline = SentinelFeaturePipeline.load(PIPELINE_PATH)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    with open(AUDIT_REPORT_PATH, "r") as f:
        audit = json.load(f)
    return {
        "test_df": test_df,
        "pipeline": pipeline,
        "model": model,
        "threshold": config["validation_metrics"]["optimal_threshold"],
        "audit": audit
    }


def test_exact_metrics_reproduction(audit_data):
    """Verify recalculation matches the audit report exactly without numerical drift."""
    test_df = audit_data["test_df"]
    pipeline = audit_data["pipeline"]
    model = audit_data["model"]
    threshold = audit_data["threshold"]
    
    X_test = pipeline.transform(test_df)
    y_test = test_df["isFraud"].values.astype(np.int8)
    probs = model.predict_proba(X_test)[:, 1]
    
    recalc_pr_auc = float(average_precision_score(y_test, probs))
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    
    expected = audit_data["audit"]["reproduction_summary"]
    assert abs(recalc_pr_auc - expected["reproduced_pr_auc"]) < 1e-4
    assert tn == expected["reproduced_confusion_matrix"]["TN"]
    assert fp == expected["reproduced_confusion_matrix"]["FP"]
    assert fn == expected["reproduced_confusion_matrix"]["FN"]
    assert tp == expected["reproduced_confusion_matrix"]["TP"]


def test_subgroup_slice_fidelity(audit_data):
    """Verify TRANSFER and CASH_OUT slice metrics match audit expectations."""
    test_df = audit_data["test_df"]
    pipeline = audit_data["pipeline"]
    model = audit_data["model"]
    threshold = audit_data["threshold"]
    
    X_test = pipeline.transform(test_df)
    y_test = test_df["isFraud"].values.astype(np.int8)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    # TRANSFER slice
    transfer_mask = (test_df["type"] == "TRANSFER").values
    transfer_y = y_test[transfer_mask]
    transfer_probs = probs[transfer_mask]
    transfer_pr_auc = float(average_precision_score(transfer_y, transfer_probs))
    assert transfer_pr_auc >= 0.995, f"TRANSFER PR-AUC degraded: {transfer_pr_auc:.4f}"
    
    # CASH_OUT slice
    cashout_mask = (test_df["type"] == "CASH_OUT").values
    cashout_y = y_test[cashout_mask]
    cashout_probs = probs[cashout_mask]
    cashout_pr_auc = float(average_precision_score(cashout_y, cashout_probs))
    assert cashout_pr_auc >= 0.999, f"CASH_OUT PR-AUC degraded: {cashout_pr_auc:.4f}"


def test_zero_false_positives_in_zero_fraud_types(audit_data):
    """Verify model produces zero false positives in transaction types with 0 fraud."""
    test_df = audit_data["test_df"]
    pipeline = audit_data["pipeline"]
    model = audit_data["model"]
    threshold = audit_data["threshold"]
    
    zero_fraud_mask = test_df["type"].isin(["PAYMENT", "CASH_IN", "DEBIT"]).values
    zero_df = test_df[zero_fraud_mask]
    X_zero = pipeline.transform(zero_df)
    probs_zero = model.predict_proba(X_zero)[:, 1]
    preds_zero = (probs_zero >= threshold).astype(int)
    
    assert preds_zero.sum() == 0, f"False positives detected in non-fraud types: {preds_zero.sum()}"


def test_no_account_identifiers_in_features():
    """Verify raw account string identifiers are strictly excluded from predictors."""
    for feat in FEATURE_NAMES:
        assert feat not in ["nameOrig", "nameDest"], f"Account ID leaked into features: {feat}"


def test_no_duplicate_records_in_test_set(audit_data):
    """Verify holdout test set has zero exact duplicate transactions."""
    test_df = audit_data["test_df"]
    duplicates = test_df.duplicated(subset=["step", "type", "amount", "nameOrig", "nameDest"]).sum()
    assert duplicates == 0, f"Duplicate transactions found in test set: {duplicates}"


def test_ablation_hierarchy_integrity(audit_data):
    """Verify information gradient: Full Model A > No-Proxy Model B > No-Balance Model C."""
    abl = audit_data["audit"]["ablation_study"]
    pr_a = abl["model_a_production"]["pr_auc"]
    pr_b = abl["model_b_no_depletion_proxies"]["pr_auc"]
    pr_c = abl["model_c_no_balances"]["pr_auc"]
    
    assert pr_a > pr_b > pr_c, f"Ablation hierarchy violated: A={pr_a}, B={pr_b}, C={pr_c}"
