"""
Automated Test Suite for Phase 2: Feature Engineering, Leakage Auditing & XGBoost Baseline Model
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys_path = str(PROJECT_ROOT)
import sys
if sys_path not in sys.path:
    sys.path.append(sys_path)

from ml.features.feature_pipeline import SentinelFeaturePipeline, FEATURE_NAMES, EXCLUDED_TARGET_FIELDS

MODELS_DIR = PROJECT_ROOT / "ml" / "models"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

MODEL_PATH = MODELS_DIR / "xgboost_baseline.json"
PIPELINE_PATH = MODELS_DIR / "feature_pipeline.joblib"
CONFIG_PATH = MODELS_DIR / "model_config.json"
TEST_DATA_PATH = PROCESSED_DIR / "test.parquet"

@pytest.fixture(scope="module")
def loaded_pipeline():
    assert PIPELINE_PATH.exists(), f"Feature pipeline artifact missing at {PIPELINE_PATH}"
    return SentinelFeaturePipeline.load(PIPELINE_PATH)

@pytest.fixture(scope="module")
def loaded_model():
    assert MODEL_PATH.exists(), f"XGBoost model artifact missing at {MODEL_PATH}"
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

@pytest.fixture(scope="module")
def sample_test_batch():
    assert TEST_DATA_PATH.exists(), f"Test data missing at {TEST_DATA_PATH}"
    df = pd.read_parquet(TEST_DATA_PATH)
    return df.head(100)


def test_feature_schema_consistency(loaded_pipeline, sample_test_batch):
    """Verify transformed feature matrix matches the expected schema and ordering."""
    features_df = loaded_pipeline.transform(sample_test_batch)
    assert list(features_df.columns) == FEATURE_NAMES
    assert len(features_df) == len(sample_test_batch)
    assert features_df.isnull().sum().sum() == 0, "Transformed features contain NaNs"
    assert features_df.dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all(), "All features must be numeric floats"


def test_leakage_exclusion_in_features(loaded_pipeline, sample_test_batch):
    """Verify that isFraud and isFlaggedFraud are strictly excluded from transformed predictors."""
    features_df = loaded_pipeline.transform(sample_test_batch)
    for excluded in EXCLUDED_TARGET_FIELDS:
        assert excluded not in features_df.columns, f"Leakage detected: {excluded} found in features!"
        assert excluded not in loaded_pipeline.feature_names_


def test_pre_authorization_compliance(sample_test_batch):
    """Verify that pipeline functions correctly without post-settlement columns."""
    pre_auth_df = sample_test_batch.drop(columns=["newbalanceOrig", "newbalanceDest", "isFraud"], errors="ignore")
    pipeline = SentinelFeaturePipeline(mode="pre_auth")
    features_df = pipeline.transform(pre_auth_df)
    assert len(features_df) == len(pre_auth_df)
    assert list(features_df.columns) == FEATURE_NAMES


def test_deterministic_preprocessing(loaded_pipeline, sample_test_batch):
    """Verify preprocessing is strictly deterministic on repeated calls."""
    run1 = loaded_pipeline.transform(sample_test_batch).values
    run2 = loaded_pipeline.transform(sample_test_batch).values
    np.testing.assert_array_equal(run1, run2, err_msg="Preprocessing is non-deterministic")


def test_valid_probability_outputs(loaded_model, loaded_pipeline, sample_test_batch):
    """Verify predicted probabilities are within valid range [0.0, 1.0]."""
    X = loaded_pipeline.transform(sample_test_batch)
    probs = loaded_model.predict_proba(X)
    assert probs.shape == (len(sample_test_batch), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities outside [0, 1]"
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5, err_msg="Probabilities must sum to 1.0")


def test_model_serialization_and_reloading(loaded_model, loaded_pipeline, sample_test_batch):
    """Verify reloaded model produces identical outputs to original in-memory model."""
    reloaded_model = xgb.XGBClassifier()
    reloaded_model.load_model(MODEL_PATH)
    
    X = loaded_pipeline.transform(sample_test_batch)
    preds1 = loaded_model.predict_proba(X)
    preds2 = reloaded_model.predict_proba(X)
    
    np.testing.assert_array_almost_equal(preds1, preds2, decimal=6)


def test_representative_single_inference(loaded_model, loaded_pipeline):
    """Verify single dictionary inference pipeline with realistic fraud and legitimate payloads."""
    # 1. Obvious fraud scenario: TRANSFER draining full balance to account
    fraud_payload = {
        "step": 650,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C199999999",
        "oldbalanceOrg": 250000.0,
        "nameDest": "C888888888",
        "oldbalanceDest": 0.0
    }
    vec_fraud = loaded_pipeline.transform_single(fraud_payload)
    prob_fraud = float(loaded_model.predict_proba(vec_fraud)[0, 1])
    
    # 2. Obvious legitimate scenario: Small merchant PAYMENT with ample balance
    legit_payload = {
        "step": 650,
        "type": "PAYMENT",
        "amount": 45.50,
        "nameOrig": "C111111111",
        "oldbalanceOrg": 8500.0,
        "nameDest": "M999999999",
        "oldbalanceDest": 0.0
    }
    vec_legit = loaded_pipeline.transform_single(legit_payload)
    prob_legit = float(loaded_model.predict_proba(vec_legit)[0, 1])
    
    # The operational fraud threshold is 0.620 (Low risk tier is < 0.40)
    assert prob_fraud >= 0.620, f"Fraud transaction scored below operational threshold: {prob_fraud:.4f}"
    assert prob_legit < 0.40, f"Legitimate transaction scored too high: {prob_legit:.4f}"
    assert prob_fraud > prob_legit + 0.25, "Fraud and legitimate scores did not have sufficient separation"
