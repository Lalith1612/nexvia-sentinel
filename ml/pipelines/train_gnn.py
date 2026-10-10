"""
Training Pipeline for Sentinel PyTorch Geometric GraphSAGE Model
Trains an inductive edge-conditioned GNN for financial fraud and money mule detection.
Employs class-imbalance weighting, validation PR-AUC monitoring, and resource profiling.
"""

import json
import os
import sys
import time
from pathlib import Path
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ml.features.graph_builder import SentinelGraphBuilder, NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES
from ml.models.gnn_model import SentinelGraphSAGE

PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROCESSED_DIR / "train_benchmark_sample.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"

MODEL_SAVE_PATH = MODELS_DIR / "gnn_graphsage.pt"
MAPPING_SAVE_PATH = MODELS_DIR / "graph_node_mapping.json"
CONFIG_SAVE_PATH = MODELS_DIR / "gnn_config.json"

def train_gnn():
    start_time = time.perf_counter()
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / (1024 * 1024)
    print("=" * 80)
    print("SENTINEL PYTORCH GEOMETRIC GNN TRAINING PIPELINE")
    print(f"Initial Process Memory: {initial_mem_mb:.2f} MB")
    print("=" * 80)

    # 1. Load Parquet datasets
    print("\n[TRAIN_GNN 1/5] Loading train and validation partitions...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    print(f"  Train Transactions : {len(train_df):,} ({int((train_df['isFraud']==1).sum()):,} frauds)")
    print(f"  Val Transactions   : {len(val_df):,} ({int((val_df['isFraud']==1).sum()):,} frauds)")

    # 2. Build Graphs
    print("\n[TRAIN_GNN 2/5] Building temporal graph structures...")
    builder = SentinelGraphBuilder()
    train_data, train_meta = builder.build_graph_data(train_df, is_training=True)
    val_data, val_meta = builder.build_graph_data(val_df, is_training=False)

    post_graph_mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"  Graph Memory Usage : {post_graph_mem_mb - initial_mem_mb:.2f} MB (Total: {post_graph_mem_mb:.2f} MB)")

    # 3. Model & Loss Setup
    print("\n[TRAIN_GNN 3/5] Initializing SentinelGraphSAGE model...")
    device = torch.device("cpu")
    print(f"  Compute Device: {device}")

    model = SentinelGraphSAGE(
        node_in_channels=len(NODE_FEATURE_NAMES),
        edge_in_channels=len(EDGE_FEATURE_NAMES),
        hidden_channels=64,
        out_channels=32,
        dropout=0.2
    ).to(device)

    # Class imbalance weighting
    n_pos = train_meta["fraud_edges"]
    n_neg = train_meta["legit_edges"]
    pos_weight_val = float(n_neg / max(1, n_pos))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    print(f"  Computed pos_weight for BCE Loss: {pos_weight_val:.2f}")

    # Move tensors to device
    train_x = train_data.x.to(device)
    train_edge_index = train_data.edge_index.to(device)
    train_edge_attr = train_data.edge_attr.to(device)
    train_y = train_data.y.to(device)

    val_x = val_data.x.to(device)
    val_edge_index = val_data.edge_index.to(device)
    val_edge_attr = val_data.edge_attr.to(device)
    val_y_np = val_data.y.numpy()

    # 4. Training Loop with Validation Monitoring
    print("\n[TRAIN_GNN 4/5] Training for 25 epochs with PR-AUC validation tracking...")
    num_epochs = 25
    best_val_pr_auc = 0.0
    best_epoch = 0
    patience = 8
    patience_counter = 0

    epoch_history = []

    for epoch in range(1, num_epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        # Forward pass on training graph
        logits = model(train_x, train_edge_index, train_edge_attr)
        loss = criterion(logits, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_probs = model.predict_proba(val_x, val_edge_index, val_edge_attr).numpy()
            val_pr_auc = float(average_precision_score(val_y_np, val_probs))

        t_epoch = time.perf_counter() - t_epoch_start
        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(float(loss.item()), 4),
            "val_pr_auc": round(val_pr_auc, 4),
            "epoch_time_s": round(t_epoch, 2)
        })

        if epoch % 5 == 0 or epoch == 1 or val_pr_auc > best_val_pr_auc:
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {loss.item():.4f} | Val PR-AUC: {val_pr_auc:.4f} | Time: {t_epoch:.2f}s")

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (best epoch: {best_epoch} with Val PR-AUC: {best_val_pr_auc:.4f})")
                break

    # 5. Load best model and tune validation threshold
    print("\n[TRAIN_GNN 5/5] Tuning operational decision threshold on validation set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    with torch.no_grad():
        final_val_probs = model.predict_proba(val_x, val_edge_index, val_edge_attr).numpy()

    best_thresh = 0.50
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for th in np.linspace(0.10, 0.95, 86):
        preds = (final_val_probs >= th).astype(int)
        f = f1_score(val_y_np, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = float(th)
            best_prec = float(precision_score(val_y_np, preds, zero_division=0))
            best_rec = float(recall_score(val_y_np, preds, zero_division=0))

    peak_mem_mb = process.memory_info().rss / (1024 * 1024)
    total_time_s = time.perf_counter() - start_time

    print(f"  Optimal Val Threshold : {best_thresh:.3f}")
    print(f"  Val F1 at Threshold    : {best_f1:.4f} (Prec: {best_prec*100:.2f}%, Rec: {best_rec*100:.2f}%)")
    print(f"  Peak RAM Consumption  : {peak_mem_mb:.2f} MB (Delta: {peak_mem_mb - initial_mem_mb:.2f} MB)")
    print(f"  Total Runtime          : {total_time_s:.2f} seconds")

    # Save artifacts
    builder.save_mapping(MAPPING_SAVE_PATH)
    print(f"  Node mapping saved to : {MAPPING_SAVE_PATH}")
    print(f"  Model weights saved to: {MODEL_SAVE_PATH}")

    config = {
        "model_architecture": "SentinelGraphSAGE",
        "node_features": NODE_FEATURE_NAMES,
        "edge_features": EDGE_FEATURE_NAMES,
        "hyperparameters": {
            "hidden_channels": 64,
            "out_channels": 32,
            "dropout": 0.2,
            "lr": 0.005,
            "pos_weight": round(pos_weight_val, 2),
            "best_epoch": best_epoch,
            "total_epochs_trained": len(epoch_history)
        },
        "validation_performance": {
            "val_pr_auc": round(best_val_pr_auc, 4),
            "optimal_threshold": round(best_thresh, 3),
            "val_f1": round(best_f1, 4),
            "val_precision": round(best_prec, 4),
            "val_recall": round(best_rec, 4)
        },
        "resource_metrics": {
            "peak_memory_mb": round(peak_mem_mb, 2),
            "net_memory_delta_mb": round(peak_mem_mb - initial_mem_mb, 2),
            "training_runtime_seconds": round(total_time_s, 2)
        },
        "epoch_history": epoch_history
    }

    with open(CONFIG_SAVE_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Model configuration saved to: {CONFIG_SAVE_PATH}")
    print("=" * 80)
    print("SENTINEL GNN TRAINING COMPLETE")
    print("=" * 80)
    return config

if __name__ == "__main__":
    train_gnn()
