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


def test_graph_builder_node_indexing(sample_tx_batch):
    """Verify contiguous integer node indexing and feature matrix alignment."""
    builder = SentinelGraphBuilder()
    graph_data, meta = builder.build_graph_data(sample_tx_batch)

    assert graph_data.num_nodes == meta["num_nodes"]
    assert graph_data.x.shape == (meta["num_nodes"], len(NODE_FEATURE_NAMES))
    assert graph_data.edge_index.shape == (2, len(sample_tx_batch))
    assert graph_data.edge_attr.shape == (len(sample_tx_batch), len(EDGE_FEATURE_NAMES))
    assert graph_data.y.shape == (len(sample_tx_batch),)


def test_temporal_edge_sorting(sample_tx_batch):
    """Verify that graph construction preserves chronological step ordering."""
    builder = SentinelGraphBuilder()
    graph_data, _ = builder.build_graph_data(sample_tx_batch)
    steps = graph_data.edge_step.numpy()
    assert np.all(np.diff(steps) >= 0), "Graph edges violate chronological step ordering"


def test_zero_label_leakage_in_graph(sample_tx_batch):
    """Verify that isFraud is strictly excluded from node and edge feature tensors."""
    builder = SentinelGraphBuilder()
    graph_data, _ = builder.build_graph_data(sample_tx_batch)

    assert "isFraud" not in NODE_FEATURE_NAMES
    assert "isFraud" not in EDGE_FEATURE_NAMES
    # Target label y must be separated
    assert not torch.isnan(graph_data.y).any()


def test_gnn_model_architecture_shapes(gnn_model, sample_tx_batch):
    """Verify forward and predict_proba produce expected output shapes."""
    builder = SentinelGraphBuilder()
    graph_data, _ = builder.build_graph_data(sample_tx_batch)

    with torch.no_grad():
        logits = gnn_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        probs = gnn_model.predict_proba(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

    assert logits.shape == (len(sample_tx_batch),)
    assert probs.shape == (len(sample_tx_batch),)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_gnn_account_risk_scoring(gnn_model, sample_tx_batch):
    """Verify account-level risk aggregation returns valid scores for all accounts."""
    builder = SentinelGraphBuilder()
    graph_data, _ = builder.build_graph_data(sample_tx_batch)

    risk_scores = gnn_model.get_account_risk_scores(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    assert risk_scores.shape == (graph_data.num_nodes,)
    assert (risk_scores >= 0.0).all() and (risk_scores <= 1.0).all()


def test_gnn_model_reproducibility(gnn_model, sample_tx_batch):
    """Verify identical outputs on repeated evaluations."""
    builder = SentinelGraphBuilder()
    graph_data, _ = builder.build_graph_data(sample_tx_batch)

    with torch.no_grad():
        run1 = gnn_model.predict_proba(graph_data.x, graph_data.edge_index, graph_data.edge_attr).numpy()
        run2 = gnn_model.predict_proba(graph_data.x, graph_data.edge_index, graph_data.edge_attr).numpy()

    np.testing.assert_array_equal(run1, run2)
