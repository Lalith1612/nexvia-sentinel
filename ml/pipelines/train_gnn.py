"""
Training Pipeline for Sentinel PyTorch Geometric GraphSAGE Model (Temporal Leakage-Free)
Trains an inductive edge-conditioned GNN for financial fraud detection.
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
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

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

def prepare_causal_training_windows(
    train_df: pd.DataFrame,
    builder: SentinelGraphBuilder = None,
    window_size: int = 25,
    max_steps: int = 500
):
    """
    Constructs chronological training windows enforcing strict causal ordering:
    1. Candidate edge registration & pre-authorization edge features without state update.
    2. Historical node features derived strictly from counters settled prior to current window.
    3. Transactions settled into history only AFTER capturing prior window features.
    """
    if builder is None:
        builder = SentinelGraphBuilder()

    num_windows = max_steps // window_size
    windows = [(i * window_size + 1, (i + 1) * window_size) for i in range(num_windows)]
    window_data = []
    hist_edges = torch.empty((2, 0), dtype=torch.long)
    total_fraud_edges = 0
    total_legit_edges = 0

    for w_idx, (s_start, s_end) in enumerate(windows):
        df_w = train_df[(train_df["step"] >= s_start) & (train_df["step"] <= s_end)]
        if len(df_w) == 0:
            continue
        # 1. Candidate edge registration & pre-authorization features without state update
        q_edge, q_attr, y_w, _ = builder.build_step_candidate_edges(df_w, update_state=False)
        # 2. Historical node features strictly prior to window w
        x_w = builder.get_current_node_features()
        # 3. Store window data: (x prior to w, edges settled prior to w, candidate query edges, candidate edge attrs, labels)
        window_data.append((x_w, hist_edges.clone(), q_edge, q_attr, y_w))
        # 4. Settle transactions into history strictly after window state is captured
        builder.settle_step_transactions(q_edge, df_w["amount"].values)
        hist_edges = torch.cat([hist_edges, q_edge], dim=1)
        total_fraud_edges += int((y_w == 1.0).sum())
        total_legit_edges += int((y_w == 0.0).sum())

    return window_data, builder, hist_edges, total_fraud_edges, total_legit_edges


def train_gnn():
    start_time = time.perf_counter()
    process = psutil.Process()
    initial_mem_mb = process.memory_info().rss / (1024 * 1024)

    print("=" * 80)
    print("SENTINEL PYTORCH GEOMETRIC GNN TRAINING PIPELINE (LEAKAGE-FREE)")
    print(f"Initial Process Memory: {initial_mem_mb:.2f} MB")
    print("=" * 80)

    # 1. Load Parquet datasets
    print("\n[TRAIN_GNN 1/5] Loading train and validation partitions...")
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    print(f"  Train Transactions : {len(train_df):,} ({int((train_df['isFraud']==1).sum()):,} frauds)")
    print(f"  Val Transactions   : {len(val_df):,} ({int((val_df['isFraud']==1).sum()):,} frauds)")

    # 2. Precompute 20 Causal Temporal Training Windows (Steps 1-500)
    print("\n[TRAIN_GNN 2/5] Constructing 20 chronological causal training windows without future leakage...")
    t_pre_start = time.perf_counter()
    window_data, builder, hist_edges, total_fraud_edges, total_legit_edges = prepare_causal_training_windows(
        train_df=train_df, window_size=25, max_steps=500
    )
    print(f"  Precomputed {len(window_data)} causal windows in {time.perf_counter() - t_pre_start:.2f}s")
    print(f"  Indexed {len(builder.idx_to_node):,} unique training accounts across steps 1-500.")

    # Validation Graph Setup: inherits settled training history (steps 1-500)
    val_data, val_aligned_df, val_meta = builder.build_graph_data(val_df, update_state=False)
    val_x = val_data.x
    val_edge_index = val_data.edge_index
    val_edge_attr = val_data.edge_attr
    val_y_np = val_data.y.numpy()

    # 3. Model & Loss Setup
    print("\n[TRAIN_GNN 3/5] Initializing SentinelGraphSAGE model for causal temporal training...")
    device = torch.device("cpu")

    model = SentinelGraphSAGE(
        node_in_channels=len(NODE_FEATURE_NAMES),
        edge_in_channels=len(EDGE_FEATURE_NAMES),
        hidden_channels=64,
        out_channels=32,
        dropout=0.2
    ).to(device)

    pos_weight_val = float(total_legit_edges / max(1, total_fraud_edges))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    print(f"  Computed pos_weight for BCE Loss: {pos_weight_val:.2f}")

    # 4. Training Loop with Causal Temporal Windows & Validation Monitoring
    print("\n[TRAIN_GNN 4/5] Training for up to 20 epochs across causal windows with early stopping...")
    num_epochs = 20
    best_val_pr_auc = 0.0
    best_epoch = 0
    patience = 6
    patience_counter = 0

    epoch_history = []

    for epoch in range(1, num_epochs + 1):
        t_epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_w, e_w, q_edge, q_attr, y_w in window_data:
            optimizer.zero_grad()
            # Causal forward pass: message passing strictly on edges settled prior to window (e_w)
            dummy_attr = torch.empty((e_w.shape[1], len(EDGE_FEATURE_NAMES)), dtype=torch.float)
            logits = model(x_w, e_w, dummy_attr, query_edge_index=q_edge, query_edge_attr=q_attr)
            loss = criterion(logits, y_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)

        # Validation evaluation using historical training edges for message passing
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x, hist_edges, torch.empty((hist_edges.shape[1], len(EDGE_FEATURE_NAMES))),
                               query_edge_index=val_edge_index, query_edge_attr=val_edge_attr)
            val_probs = torch.sigmoid(val_logits).numpy()
            val_pr_auc = float(average_precision_score(val_y_np, val_probs))

        t_epoch = time.perf_counter() - t_epoch_start
        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_pr_auc": round(val_pr_auc, 4),
            "epoch_time_s": round(t_epoch, 2)
        })

        if epoch % 2 == 0 or epoch == 1 or val_pr_auc > best_val_pr_auc:
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {avg_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | Time: {t_epoch:.2f}s")

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
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
        dummy_hist_attr = torch.empty((hist_edges.shape[1], len(EDGE_FEATURE_NAMES)), dtype=torch.float)
        final_val_logits = model(val_x, hist_edges, dummy_hist_attr, query_edge_index=val_edge_index, query_edge_attr=val_edge_attr)
        final_val_probs = torch.sigmoid(final_val_logits).numpy()

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

    # Save state
    builder.save_state(MAPPING_SAVE_PATH)
    print(f"  Node state saved to    : {MAPPING_SAVE_PATH}")
    print(f"  Model weights saved to : {MODEL_SAVE_PATH}")

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
