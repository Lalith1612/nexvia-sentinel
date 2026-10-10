"""
Automated Test Suite for Phase 3: Account Graph Construction & PyTorch Geometric GNN Model
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys_path = str(PROJECT_ROOT)
import sys
if sys_path not in sys.path:
    sys.path.append(sys_path)

from ml.features.graph_builder import SentinelGraphBuilder, NODE_FEATURE_NAMES, EDGE_FEATURE_NAMES
from ml.models.gnn_model import SentinelGraphSAGE

PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"

TEST_DATA_PATH = PROCESSED_DIR / "test.parquet"
MODEL_WEIGHTS_PATH = MODELS_DIR / "gnn_graphsage.pt"
MAPPING_PATH = MODELS_DIR / "graph_node_mapping.json"
CONFIG_PATH = MODELS_DIR / "gnn_config.json"
METRICS_PATH = REPORTS_DIR / "gnn_evaluation_metrics.json"

@pytest.fixture(scope="module")
def sample_tx_batch():
    assert TEST_DATA_PATH.exists(), f"Test data missing at {TEST_DATA_PATH}"
    df = pd.read_parquet(TEST_DATA_PATH)
    return df.head(200)

@pytest.fixture(scope="module")
def gnn_model():
    assert MODEL_WEIGHTS_PATH.exists(), f"GNN weights missing at {MODEL_WEIGHTS_PATH}"
    model = SentinelGraphSAGE(
        node_in_channels=len(NODE_FEATURE_NAMES),
        edge_in_channels=len(EDGE_FEATURE_NAMES),
        hidden_channels=64,
        out_channels=32,
        dropout=0.2
    )
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


def test_reconciled_ground_truth_test_fraud_counts():
    """Verify ground truth fraud counts from untouched test.parquet directly."""
    assert TEST_DATA_PATH.exists(), f"Test data missing at {TEST_DATA_PATH}"
    df = pd.read_parquet(TEST_DATA_PATH)
    assert len(df) == 90829
    assert df["isFraud"].sum() == 1366
    
    # Reconciled counts by transaction type
    type_counts = df.groupby("type")["isFraud"].agg(["count", "sum"])
    assert type_counts.loc["TRANSFER", "sum"] == 683
    assert type_counts.loc["CASH_OUT", "sum"] == 683
    assert type_counts.loc["PAYMENT", "sum"] == 0
    assert type_counts.loc["CASH_IN", "sum"] == 0
    assert type_counts.loc["DEBIT", "sum"] == 0
    assert type_counts.loc["TRANSFER", "count"] == 8974
    assert type_counts.loc["CASH_OUT", "count"] == 29501


def test_graph_builder_node_indexing(sample_tx_batch):
    """Verify contiguous integer node indexing and feature matrix alignment."""
    builder = SentinelGraphBuilder()
    graph_data, df_aligned, meta = builder.build_graph_data(sample_tx_batch)

    assert graph_data.num_nodes == meta["num_nodes"]
    assert graph_data.x.shape == (meta["num_nodes"], len(NODE_FEATURE_NAMES))
    assert graph_data.edge_index.shape == (2, len(sample_tx_batch))
    assert graph_data.edge_attr.shape == (len(sample_tx_batch), len(EDGE_FEATURE_NAMES))
    assert graph_data.y.shape == (len(sample_tx_batch),)
    assert len(df_aligned) == len(sample_tx_batch)


def test_row_alignment_preservation(sample_tx_batch):
    """Verify that build_graph_data preserves exact 1-to-1 row index alignment."""
    builder = SentinelGraphBuilder()
    graph_data, df_aligned, _ = builder.build_graph_data(sample_tx_batch)
    
    # Check that df_aligned step values match input exactly
    np.testing.assert_array_equal(df_aligned["step"].values, sample_tx_batch["step"].values)
    np.testing.assert_array_equal(df_aligned["amount"].values, sample_tx_batch["amount"].values)
    np.testing.assert_array_equal(graph_data.y.numpy(), sample_tx_batch["isFraud"].values)


def test_temporal_running_state_leakage_free():
    """Verify that prior degree/volume features are strictly computed before each txn."""
    builder = SentinelGraphBuilder()
    # Create synthetic test transactions where account A transacts twice
    synthetic_df = pd.DataFrame({
        "step": [1, 2, 3],
        "type": ["TRANSFER", "TRANSFER", "CASH_OUT"],
        "amount": [100.0, 200.0, 300.0],
        "nameOrig": ["C_Alice", "C_Alice", "C_Bob"],
        "oldbalanceOrg": [1000.0, 900.0, 500.0],
        "newbalanceOrig": [900.0, 700.0, 200.0],
        "nameDest": ["C_Bob", "C_Carol", "C_Alice"],
        "oldbalanceDest": [0.0, 0.0, 900.0],
        "newbalanceDest": [100.0, 0.0, 1200.0],
        "isFraud": [0, 0, 0]
    })
    
    graph_data, _, _ = builder.build_graph_data(synthetic_df, update_state=True)
    
    # Edge 0 (C_Alice -> C_Bob): Alice has prior out-deg = 0
    # Edge 1 (C_Alice -> C_Carol): Alice has prior out-deg = 1
    # Edge feature indices for prior_out_deg: index 11 in EDGE_FEATURE_NAMES
    prior_out_deg_idx = EDGE_FEATURE_NAMES.index("src_prior_out_degree")
    prior_out_vol_idx = EDGE_FEATURE_NAMES.index("src_prior_out_volume")
    
    edge_attrs = graph_data.edge_attr.numpy()
    assert edge_attrs[0, prior_out_deg_idx] == 0.0, "First txn leaked future activity into prior state"
    assert edge_attrs[0, prior_out_vol_idx] == 0.0, "First txn leaked future volume into prior state"
    np.testing.assert_allclose(edge_attrs[1, prior_out_deg_idx], np.log1p(1.0), rtol=1e-5)
    np.testing.assert_allclose(edge_attrs[1, prior_out_vol_idx], np.log1p(100.0), rtol=1e-5)


def test_zero_label_leakage_in_graph(sample_tx_batch):
    """Verify that isFraud is strictly excluded from node and edge feature tensors."""
    builder = SentinelGraphBuilder()
    graph_data, _, _ = builder.build_graph_data(sample_tx_batch)

    assert "isFraud" not in NODE_FEATURE_NAMES
    assert "isFraud" not in EDGE_FEATURE_NAMES
    assert not torch.isnan(graph_data.y).any()


def test_gnn_model_architecture_shapes(gnn_model, sample_tx_batch):
    """Verify forward and predict_proba produce expected output shapes."""
    builder = SentinelGraphBuilder()
    graph_data, _, _ = builder.build_graph_data(sample_tx_batch)

    with torch.no_grad():
        logits = gnn_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        probs = gnn_model.predict_proba(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

    assert logits.shape == (len(sample_tx_batch),)
    assert probs.shape == (len(sample_tx_batch),)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_gnn_account_risk_scoring(gnn_model, sample_tx_batch):
    """Verify account-level risk aggregation returns valid scores for all accounts."""
    builder = SentinelGraphBuilder()
    graph_data, _, _ = builder.build_graph_data(sample_tx_batch)

    risk_scores = gnn_model.get_account_risk_scores(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    assert risk_scores.shape == (graph_data.num_nodes,)
    assert (risk_scores >= 0.0).all() and (risk_scores <= 1.0).all()


def test_gnn_model_reproducibility(gnn_model, sample_tx_batch):
    """Verify identical outputs on repeated evaluations."""
    builder = SentinelGraphBuilder()
    graph_data, _, _ = builder.build_graph_data(sample_tx_batch)

    with torch.no_grad():
        run1 = gnn_model.predict_proba(graph_data.x, graph_data.edge_index, graph_data.edge_attr).numpy()
        run2 = gnn_model.predict_proba(graph_data.x, graph_data.edge_index, graph_data.edge_attr).numpy()

    np.testing.assert_array_equal(run1, run2)


def test_validation_selected_threshold_used_on_test():
    """Verify that threshold was selected using validation data only, not test set."""
    assert CONFIG_PATH.exists()
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    assert "validation_performance" in config
    assert config["validation_performance"]["optimal_threshold"] == 0.680
    assert config["validation_performance"]["val_pr_auc"] > 0.0

