"""
XGBoost Baseline Model Training Pipeline for Sentinel (Phase 2 with MLflow Tracking)
Trains an enterprise-grade gradient-boosted decision tree classifier for real-time fraud detection.
Uses class imbalance compensation (scale_pos_weight), temporal validation threshold tuning,
and automated local MLflow experiment tracking with graceful degradation.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.feature_pipeline import SentinelFeaturePipeline

logger = logging.getLogger("sentinel.ml.train_xgboost")

PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED_DIR / "train_benchmark_sample.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"


def train_xgboost(
    train_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
    output_model_path: Optional[Path] = None,
    output_config_path: Optional[Path] = None,
    output_pipeline_path: Optional[Path] = None,
    output_feature_names_path: Optional[Path] = None,
    enable_mlflow: bool = True,
    mlflow_experiment_name: str = "sentinel-fraud-detection",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
    n_estimators: int = 350,
    max_depth: int = 6,
    learning_rate: float = 0.08,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    sample_fraction: Optional[float] = None,
    save_production_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Trains XGBoost baseline fraud detection model with PR-AUC optimization
    and optional, safely configurable MLflow experiment logging.
    """
    train_file = Path(train_path) if train_path else TRAIN_PATH
    val_file = Path(val_path) if val_path else VAL_PATH

    print(f"[TRAIN] Loading training and validation partitions ({train_file.name}, {val_file.name})...")
    if not train_file.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_file}")

    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)

    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        print(f"[TRAIN] Sampling partitions with fraction={sample_fraction:.2f} for fast execution/testing...")
        train_df = train_df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

    print(f"  Training Set   : {len(train_df):,} records ({int((train_df['isFraud']==1).sum()):,} frauds, {train_df['isFraud'].mean()*100:.2f}% fraud rate)")
    print(f"  Validation Set : {len(val_df):,} records ({int((val_df['isFraud']==1).sum()):,} frauds, {val_df['isFraud'].mean()*100:.2f}% fraud rate)")

    # 1. Feature Engineering
    print("[TRAIN] Fitting feature pipeline and transforming matrices...")
    pipeline = SentinelFeaturePipeline(mode="pre_auth")
    pipeline.fit(train_df)

    X_train = pipeline.transform(train_df)
    y_train = train_df["isFraud"].values.astype(np.int8)

    X_val = pipeline.transform(val_df)
    y_val = val_df["isFraud"].values.astype(np.int8)

    # Calculate empirical scale_pos_weight
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = float(n_neg / max(1, n_pos))
    print(f"[TRAIN] Class balance in train: Neg={n_neg:,}, Pos={n_pos:,}. Computed scale_pos_weight={scale_pos_weight:.2f}")

    # 2. XGBoost Model Configuration
    xgb_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "early_stopping_rounds": min(35, max(5, n_estimators // 5)),
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
    }

    print(f"[TRAIN] Initializing XGBClassifier with tree_method='hist' and early stopping...")
    model = xgb.XGBClassifier(**xgb_params)

    # Fit model with validation monitoring
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", n_estimators)
    print(f"[TRAIN] Optimal tree count selected: {best_iteration} estimators.")

    # 3. Validation Scoring and Threshold Selection
    val_probs = model.predict_proba(X_val)[:, 1]
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_roc_auc = float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 0.5
    val_brier = float(brier_score_loss(y_val, val_probs))
    print(f"[TRAIN] Validation PR-AUC: {val_pr_auc:.4f}, ROC-AUC: {val_roc_auc:.4f}, Brier: {val_brier:.6f}")

    # Grid search for optimal F1 decision threshold on validation partition
    best_thresh = 0.5
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    thresholds = np.linspace(0.10, 0.95, 86)
    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thresh = float(th)
            best_prec = float(p)
            best_rec = float(r)

    print(f"[TRAIN] Validation Threshold Tuning:")
    print(f"  Selected Optimal Threshold : {best_thresh:.3f}")
    print(f"  Validation F1-Score        : {best_f1:.4f}")
    print(f"  Validation Precision       : {best_prec:.4f}")
    print(f"  Validation Recall          : {best_rec:.4f}")

    # Build configuration output dictionary
    config = {
        "model_type": "XGBoost Classifier",
        "library_version": xgb.__version__,
        "hyperparameters": {k: v for k, v in xgb_params.items() if k not in ["eval_metric", "early_stopping_rounds"]},
        "best_iteration": int(best_iteration),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "feature_count": len(pipeline.feature_names_),
        "features": pipeline.feature_names_,
        "validation_metrics": {
            "pr_auc": round(float(val_pr_auc), 6),
            "roc_auc": round(float(val_roc_auc), 6),
            "brier_score": round(float(val_brier), 6),
            "optimal_threshold": round(float(best_thresh), 3),
            "f1_at_optimal_threshold": round(float(best_f1), 4),
            "precision_at_optimal_threshold": round(float(best_prec), 4),
            "recall_at_optimal_threshold": round(float(best_rec), 4),
        },
        "threshold_tiers": {
            "critical": 0.85,
            "high": 0.65,
            "medium": round(float(best_thresh), 3),
            "low": 0.20,
        },
    }

    # 4. Save Model Artifacts
    if save_production_artifacts:
        prod_model_path = MODELS_DIR / "xgboost_baseline.json"
        model.save_model(prod_model_path)
        print(f"[TRAIN] Production model saved to: {prod_model_path}")

        prod_pipeline_path = MODELS_DIR / "feature_pipeline.joblib"
        pipeline.save(prod_pipeline_path)
        print(f"[TRAIN] Production feature pipeline saved to: {prod_pipeline_path}")

        prod_fn_path = MODELS_DIR / "feature_names.json"
        with open(prod_fn_path, "w") as f:
            json.dump(pipeline.feature_names_, f, indent=2)

        prod_cfg_path = MODELS_DIR / "model_config.json"
        with open(prod_cfg_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[TRAIN] Production model configuration saved to: {prod_cfg_path}")

    # Handle custom output paths if caller specified non-default targets (e.g. tests)
    if output_model_path:
        out_m = Path(output_model_path)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(out_m)
        print(f"[TRAIN] Custom model saved to: {out_m}")

    if output_config_path:
        out_c = Path(output_config_path)
        out_c.parent.mkdir(parents=True, exist_ok=True)
        with open(out_c, "w") as f:
            json.dump(config, f, indent=2)

    # 5. MLflow Tracking Integration
    if enable_mlflow:
        _log_to_mlflow(
            model=model,
            config=config,
            xgb_params=xgb_params,
            best_iteration=best_iteration,
            scale_pos_weight=scale_pos_weight,
            val_pr_auc=val_pr_auc,
            val_roc_auc=val_roc_auc,
            val_brier=val_brier,
            best_thresh=best_thresh,
            best_f1=best_f1,
            best_prec=best_prec,
            best_rec=best_rec,
            experiment_name=mlflow_experiment_name,
            tracking_uri=mlflow_tracking_uri,
            run_name=mlflow_run_name,
        )

    return config


def _log_to_mlflow(
    model: xgb.XGBClassifier,
    config: Dict[str, Any],
    xgb_params: Dict[str, Any],
    best_iteration: int,
    scale_pos_weight: float,
    val_pr_auc: float,
    val_roc_auc: float,
    val_brier: float,
    best_thresh: float,
    best_f1: float,
    best_prec: float,
    best_rec: float,
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
) -> None:
    """Safely logs parameters, metrics, tags, and model artifacts to local MLflow store."""
    try:
        import mlflow
        import mlflow.xgboost

        # Default to local SQLite tracking store in project root (Windows path-safe)
        if not tracking_uri:
            mlruns_db_path = (PROJECT_ROOT / "mlruns.db").resolve()
            tracking_uri = f"sqlite:///{mlruns_db_path.as_posix()}"

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        effective_run_name = run_name or f"xgboost_baseline_{config['library_version']}"
        with mlflow.start_run(run_name=effective_run_name) as run:
            # Parameters
            mlflow.log_params({
                "n_estimators": xgb_params.get("n_estimators"),
                "max_depth": xgb_params.get("max_depth"),
                "learning_rate": xgb_params.get("learning_rate"),
                "subsample": xgb_params.get("subsample"),
                "colsample_bytree": xgb_params.get("colsample_bytree"),
                "scale_pos_weight": round(scale_pos_weight, 4),
                "best_iteration": int(best_iteration),
                "tree_method": xgb_params.get("tree_method"),
                "eval_metric": xgb_params.get("eval_metric"),
                "feature_count": config.get("feature_count", 20),
            })

            # Tags
            mlflow.set_tags({
                "dataset": "PaySim Synthetic Financial Datasets",
                "split_strategy": "Temporal non-overlapping",
                "model_type": "XGBoost Classifier",
                "target": "isFraud",
                "version": "1.0.0",
            })

            # Metrics
            mlflow.log_metrics({
                "val_pr_auc": round(float(val_pr_auc), 6),
                "val_roc_auc": round(float(val_roc_auc), 6),
                "val_brier_score": round(float(val_brier), 6),
                "val_optimal_threshold": round(float(best_thresh), 4),
                "val_f1_optimal": round(float(best_f1), 4),
                "val_precision_optimal": round(float(best_prec), 4),
                "val_recall_optimal": round(float(best_rec), 4),
            })

            # Log model artifact JSON
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp_m:
                model.save_model(tmp_m.name)
                tmp_model_path = tmp_m.name
            mlflow.log_artifact(tmp_model_path, artifact_path="model")
            Path(tmp_model_path).unlink(missing_ok=True)

            # Log configuration artifact JSON
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp_f:
                json.dump(config, tmp_f, indent=2)
                tmp_cfg_path = tmp_f.name

            mlflow.log_artifact(tmp_cfg_path, artifact_path="config")
            Path(tmp_cfg_path).unlink(missing_ok=True)

            # Also log MLflow XGBoost flavor if supported
            try:
                mlflow.xgboost.log_model(model, artifact_path="model_flavor")
            except Exception:
                pass

            config["mlflow_run_id"] = run.info.run_id
            config["mlflow_tracking_uri"] = tracking_uri
            config["mlflow_status"] = "SUCCESS"
            print(f"[TRAIN] MLflow run successfully logged: {run.info.run_id} ({tracking_uri})")

    except Exception as exc:
        print(f"[TRAIN] MLflow logging warning (continuing training without failure): {exc}")
        config["mlflow_status"] = f"FAILED_OR_SKIPPED: {str(exc)}"


if __name__ == "__main__":
    train_xgboost()
