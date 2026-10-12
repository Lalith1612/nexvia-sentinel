"""
Unit & Integration Tests for Sentinel MLflow Tracking (Phase 2)
Verifies:
1. Smoke training run logs hyperparameters, PR-AUC, ROC-AUC, Brier score, and model artifacts to MLflow.
2. Verified run, parameters, metrics, and logged model can be queried via MlflowClient.
3. Production artifacts in ml/models/ are NOT modified or overwritten by test runs.
4. Training continues uninterrupted when enable_mlflow=False.
5. Tracking connection errors or misconfigurations are handled gracefully with clear status.
"""

import os
import sys
from pathlib import Path
import pytest
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipelines.train_xgboost import train_xgboost

PROD_MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "xgboost_baseline.json"


def test_mlflow_smoke_run_and_artifact_retrieval(tmp_path):
    """
    Executes a fast, isolated smoke training workflow with MLflow enabled.
    Verifies that:
    1. The run completes with mlflow_status == 'SUCCESS'.
    2. The run ID exists and can be retrieved via MlflowClient.
    3. Parameters, metrics, tags, and model artifacts are accurately logged.
    4. The production model artifact is untouched.
    """
    # Record modification timestamp of production model if it exists
    prod_mtime_before = PROD_MODEL_PATH.stat().st_mtime if PROD_MODEL_PATH.exists() else None

    test_db = (tmp_path / "smoke_mlflow.db").resolve()
    tracking_uri = f"sqlite:///{test_db.as_posix()}"
    test_experiment = "sentinel-smoke-test-exp"
    custom_model_out = tmp_path / "custom_test_model.json"
    custom_cfg_out = tmp_path / "custom_test_config.json"

    # Fast smoke training with 1% sample fraction and 5 estimators
    config = train_xgboost(
        sample_fraction=0.01,
        n_estimators=5,
        max_depth=2,
        save_production_artifacts=False,
        output_model_path=custom_model_out,
        output_config_path=custom_cfg_out,
        enable_mlflow=True,
        mlflow_experiment_name=test_experiment,
        mlflow_tracking_uri=tracking_uri,
        mlflow_run_name="smoke_test_run",
    )

    # 1. Output verification
    assert config["mlflow_status"] == "SUCCESS"
    run_id = config.get("mlflow_run_id")
    assert run_id is not None
    assert custom_model_out.exists()
    assert custom_cfg_out.exists()

    # 2. Production artifact non-interference check
    if prod_mtime_before is not None:
        assert PROD_MODEL_PATH.stat().st_mtime == prod_mtime_before, "Production model was modified by test!"

    # 3. Retrieve and inspect run from MLflow
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    # Verify parameters
    params = run.data.params
    assert params["n_estimators"] == "5"
    assert params["max_depth"] == "2"
    assert "scale_pos_weight" in params
    assert params["tree_method"] == "hist"

    # Verify metrics
    metrics = run.data.metrics
    assert "val_pr_auc" in metrics
    assert "val_roc_auc" in metrics
    assert "val_brier_score" in metrics
    assert 0.0 <= metrics["val_pr_auc"] <= 1.0
    assert 0.0 <= metrics["val_roc_auc"] <= 1.0

    # Verify tags
    tags = run.data.tags
    assert tags.get("model_type") == "XGBoost Classifier"
    assert tags.get("dataset") == "PaySim Synthetic Financial Datasets"

    # Verify artifacts
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    assert "model" in artifacts
    assert "config" in artifacts


def test_mlflow_disabled_leaves_training_intact(tmp_path):
    """
    Verifies that when enable_mlflow=False, training completes without error
    and returns valid metrics without attempting MLflow tracking.
    """
    custom_model_out = tmp_path / "disabled_mlflow_model.json"

    config = train_xgboost(
        sample_fraction=0.01,
        n_estimators=5,
        max_depth=2,
        save_production_artifacts=False,
        output_model_path=custom_model_out,
        enable_mlflow=False,
    )

    assert "validation_metrics" in config
    assert "pr_auc" in config["validation_metrics"]
    assert "mlflow_run_id" not in config
    assert custom_model_out.exists()


def test_mlflow_tracking_failure_handled_gracefully(tmp_path):
    """
    Verifies that if MLflow encounters an invalid tracking URI or network failure,
    model training completes successfully and records a descriptive warning
    without crashing the training process.
    """
    custom_model_out = tmp_path / "resilient_model.json"

    config = train_xgboost(
        sample_fraction=0.01,
        n_estimators=5,
        max_depth=2,
        save_production_artifacts=False,
        output_model_path=custom_model_out,
        enable_mlflow=True,
        mlflow_tracking_uri="invalid_scheme://bad/uri",
        mlflow_experiment_name="failing_exp",
    )

    assert "validation_metrics" in config
    assert config["mlflow_status"].startswith("FAILED_OR_SKIPPED")
    assert custom_model_out.exists()
