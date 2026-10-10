"""
XGBoost Baseline Model Training Pipeline for Sentinel
Trains an enterprise-grade gradient-boosted decision tree classifier for real-time fraud detection.
Uses class imbalance compensation (scale_pos_weight) and temporal validation threshold tuning.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ml.features.feature_pipeline import SentinelFeaturePipeline

PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED_DIR / "train_benchmark_sample.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"

def train_xgboost():
    print(f"[TRAIN] Loading training and validation partitions...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    
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
        "n_estimators": 350,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 35,
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1
    }
    
    print(f"[TRAIN] Initializing XGBClassifier with tree_method='hist' and early stopping...")
    model = xgb.XGBClassifier(**xgb_params)
    
    # Fit model with validation monitoring
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    
    best_iteration = model.best_iteration
    print(f"[TRAIN] Optimal tree count selected: {best_iteration} estimators.")
    
    # 3. Validation Scoring and Threshold Selection
    val_probs = model.predict_proba(X_val)[:, 1]
    val_pr_auc = average_precision_score(y_val, val_probs)
    print(f"[TRAIN] Validation PR-AUC (Average Precision): {val_pr_auc:.4f}")
    
    # Grid search for optimal F1 decision threshold on validation partition
    best_thresh = 0.5
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    
    thresholds = np.linspace(0.10, 0.95, 86)
    threshold_history = []
    
    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        threshold_history.append({"threshold": round(float(th), 3), "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4)})
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            best_prec = p
            best_rec = r
            
    print(f"[TRAIN] Validation Threshold Tuning:")
    print(f"  Selected Optimal Threshold : {best_thresh:.3f}")
    print(f"  Validation F1-Score        : {best_f1:.4f}")
    print(f"  Validation Precision       : {best_prec:.4f}")
    print(f"  Validation Recall          : {best_rec:.4f}")
    
    # 4. Save Model Artifacts
    model_json_path = MODELS_DIR / "xgboost_baseline.json"
    model.save_model(model_json_path)
    print(f"[TRAIN] Model saved to: {model_json_path}")
    
    pipeline_path = MODELS_DIR / "feature_pipeline.joblib"
    pipeline.save(pipeline_path)
    print(f"[TRAIN] Feature pipeline saved to: {pipeline_path}")
    
    feature_names_path = MODELS_DIR / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(pipeline.feature_names_, f, indent=2)
        
    config = {
        "model_type": "XGBoost Classifier",
        "library_version": xgb.__version__,
        "hyperparameters": {k: v for k, v in xgb_params.items() if k not in ["eval_metric", "early_stopping_rounds"]},
        "best_iteration": int(best_iteration),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "feature_count": len(pipeline.feature_names_),
        "features": pipeline.feature_names_,
        "validation_metrics": {
            "pr_auc": round(float(val_pr_auc), 4),
            "optimal_threshold": round(float(best_thresh), 3),
            "f1_at_optimal_threshold": round(float(best_f1), 4),
            "precision_at_optimal_threshold": round(float(best_prec), 4),
            "recall_at_optimal_threshold": round(float(best_rec), 4)
        },
        "threshold_tiers": {
            "critical": 0.85,
            "high": 0.65,
            "medium": round(float(best_thresh), 3),
            "low": 0.20
        }
    }
    
    config_path = MODELS_DIR / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[TRAIN] Model configuration saved to: {config_path}")
    
    return config

if __name__ == "__main__":
    train_xgboost()
