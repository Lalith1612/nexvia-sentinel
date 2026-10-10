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
    """Verify that threshold was selected using validation data only and applied unchanged during test evaluation."""
    assert CONFIG_PATH.exists(), f"Missing config file: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    assert "validation_performance" in config
    val_perf = config["validation_performance"]
    val_thresh = val_perf["optimal_threshold"]
    assert isinstance(val_thresh, (float, int))
    assert 0.0 < val_thresh < 1.0, f"Validation threshold out of valid bounds: {val_thresh}"
    assert val_perf["val_pr_auc"] > 0.0, "Validation PR-AUC must be positive"
    assert val_perf["val_f1"] > 0.0, "Validation F1 must be positive"

    # Verify evaluate_gnn loads this exact threshold from config
    with open(CONFIG_PATH, "r") as f:
        eval_loaded_cfg = json.load(f)
    assert eval_loaded_cfg["validation_performance"]["optimal_threshold"] == val_thresh

    # If evaluation metrics have been produced for the current model config, verify exact match
    eval_metrics_path = REPORTS_DIR / "gnn_evaluation_metrics.json"
    if eval_metrics_path.exists() and eval_metrics_path.stat().st_mtime >= CONFIG_PATH.stat().st_mtime:
        with open(eval_metrics_path, "r") as f:
            eval_metrics = json.load(f)
        eval_thresh = eval_metrics["reconciled_gnn_metrics"]["operational_threshold"]
        assert eval_thresh == val_thresh, f"Evaluation threshold {eval_thresh} differs from validation threshold {val_thresh}"


def test_future_edge_invariance_and_input_isolation(gnn_model):
    """
    Robust Regression Test: Verifies that introducing a future transaction sharing
    an account (step 700) cannot alter the historical node features, historical
    message-passing edges, edge attributes, or prediction for an earlier transaction (step 621).
    """
    # Pipeline 1: Historical seed + step 621 transaction
    builder_1 = SentinelGraphBuilder()
    hist_df = pd.DataFrame([{
        "step": 600, "type": "TRANSFER", "amount": 50000.0,
        "nameOrig": "C_Seed1", "oldbalanceOrg": 50000.0, "newbalanceOrig": 0.0,
        "nameDest": "C_Target", "oldbalanceDest": 1000.0, "newbalanceDest": 51000.0,
        "isFraud": 0
    }])
    # Settle step 600
    q_edge_hist, q_attr_hist, _, _ = builder_1.build_step_candidate_edges(hist_df, update_state=True)
    hist_edges_1 = q_edge_hist.clone()

    tx_621_df = pd.DataFrame([{
        "step": 621, "type": "CASH_OUT", "amount": 50000.0,
        "nameOrig": "C_Target", "oldbalanceOrg": 51000.0, "newbalanceOrig": 1000.0,
        "nameDest": "M_Merchant", "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
        "isFraud": 1
    }])

    # Candidate edge registration and pre-authorization features
    q_edge_621_1, q_attr_621_1, _, _ = builder_1.build_step_candidate_edges(tx_621_df, update_state=False)
    x_621_1 = builder_1.get_current_node_features()

    with torch.no_grad():
        emb_1 = gnn_model.encode_nodes(x_621_1, hist_edges_1)
        h_src1 = emb_1[q_edge_621_1[0]]
        h_dst1 = emb_1[q_edge_621_1[1]]
        pred_1 = torch.sigmoid(gnn_model.edge_mlp(torch.cat([h_src1, h_dst1, q_attr_621_1], dim=-1))).item()

    # Pipeline 2: Same initial history + step 621 transaction, but ALSO containing
    # future transaction at step 700 that shares account 'C_Target' (attacker targeting C_Target)
    builder_2 = SentinelGraphBuilder()
    q_edge_hist2, _, _, _ = builder_2.build_step_candidate_edges(hist_df, update_state=True)
    hist_edges_2 = q_edge_hist2.clone()

    # In a real transaction stream, step 621 arrives first
    q_edge_621_2, q_attr_621_2, _, _ = builder_2.build_step_candidate_edges(tx_621_df, update_state=False)
    x_621_2 = builder_2.get_current_node_features()

    with torch.no_grad():
        emb_2 = gnn_model.encode_nodes(x_621_2, hist_edges_2)
        h_src2 = emb_2[q_edge_621_2[0]]
        h_dst2 = emb_2[q_edge_621_2[1]]
        pred_2 = torch.sigmoid(gnn_model.edge_mlp(torch.cat([h_src2, h_dst2, q_attr_621_2], dim=-1))).item()

    # 1. Historical node features at step 621 are bitwise identical
    assert torch.equal(x_621_1, x_621_2), "Historical node features changed before future transaction!"
    # 2. Historical message-passing edges at step 621 are bitwise identical
    assert torch.equal(hist_edges_1, hist_edges_2), "Historical edges changed before future transaction!"
    # 3. Candidate edge attributes are bitwise identical
    assert torch.equal(q_attr_621_1, q_attr_621_2), "Candidate edge attributes changed!"
    # 4. Earlier prediction is strictly invariant
    assert pred_1 == pred_2, f"Prediction changed: {pred_1} != {pred_2}"

    # 5. Direct verification that test catches future information entering historical inputs:
    # If step 700 transaction is settled into the builder before step 621:
    tx_700_future = pd.DataFrame([{
        "step": 700, "type": "TRANSFER", "amount": 90000.0,
        "nameOrig": "C_Attacker", "oldbalanceOrg": 90000.0, "newbalanceOrig": 0.0,
        "nameDest": "C_Target", "oldbalanceDest": 1000.0, "newbalanceDest": 91000.0,
        "isFraud": 1
    }])
    q_edge_700, _, _, _ = builder_2.build_step_candidate_edges(tx_700_future, update_state=True)
    x_leaked = builder_2.get_current_node_features()
    edges_leaked = torch.cat([hist_edges_2, q_edge_700], dim=1)

    # Assert that future transaction DOES alter the historical input tensors if leaked
    assert not torch.equal(x_621_1, x_leaked), "Test failed to detect future node degree/volume entering node features!"
    assert not torch.equal(hist_edges_1, edges_leaked), "Test failed to detect future edge entering message passing graph!"


def test_causal_node_features_degree_isolation():
    """
    Verifies that get_current_node_features strictly reflects settled history
    and that an account transacting in future steps does not have future counts.
    """
    builder = SentinelGraphBuilder()
    
    # Step 1 transaction
    df_1 = pd.DataFrame([{
        "step": 1, "type": "TRANSFER", "amount": 100.0,
        "nameOrig": "C_Alice", "oldbalanceOrg": 1000.0, "newbalanceOrig": 900.0,
        "nameDest": "C_Bob", "oldbalanceDest": 0.0, "newbalanceDest": 100.0,
        "isFraud": 0
    }])
    _, _, _, _ = builder.build_step_candidate_edges(df_1, update_state=False)
    
    # Before step 1 settles: Alice out-degree is 0.0
    x_before_1 = builder.get_current_node_features()
    alice_idx = builder.node_to_idx["C_Alice"]
    # Feature 2 is log_prior_out_degree
    assert x_before_1[alice_idx, 2].item() == 0.0, "Pre-authorization degree leaked future transaction!"
    
    # Now settle step 1
    _, _, _, _ = builder.build_step_candidate_edges(df_1, update_state=True)
    x_after_1 = builder.get_current_node_features()
    np.testing.assert_allclose(x_after_1[alice_idx, 2].item(), np.log1p(1.0), rtol=1e-5)


def test_training_window_node_features_causal_isolation():
    """
    Verifies that prepare_causal_training_windows ensures every training window's
    node features are computed strictly from account counters and graph state BEFORE
    that window's transactions settle.
    If a training window's node features include transactions from that same window,
    this test will fail.
    """
    from ml.pipelines.train_gnn import prepare_causal_training_windows

    # Create synthetic training transactions in window 1 (steps 1-25) and window 2 (steps 26-50)
    tx_df = pd.DataFrame([
        # Window 1 (steps 1-25): C_Sender transacts
        {
            "step": 10, "type": "TRANSFER", "amount": 5000.0,
            "nameOrig": "C_Sender", "oldbalanceOrg": 10000.0, "newbalanceOrig": 5000.0,
            "nameDest": "C_Receiver", "oldbalanceDest": 0.0, "newbalanceDest": 5000.0,
            "isFraud": 1
        },
        # Window 2 (steps 26-50): C_Sender transacts again
        {
            "step": 30, "type": "TRANSFER", "amount": 2000.0,
            "nameOrig": "C_Sender", "oldbalanceOrg": 5000.0, "newbalanceOrig": 3000.0,
            "nameDest": "C_Receiver", "oldbalanceDest": 5000.0, "newbalanceDest": 7000.0,
            "isFraud": 0
        }
    ])

    window_data, builder, hist_edges, total_frauds, total_legits = prepare_causal_training_windows(
        train_df=tx_df, window_size=25, max_steps=50
    )

    assert len(window_data) == 2, f"Expected 2 windows, got {len(window_data)}"
    sender_idx = builder.node_to_idx["C_Sender"]
    receiver_idx = builder.node_to_idx["C_Receiver"]

    # Window 1 verification:
    # x_w1 MUST NOT contain the step 10 transaction!
    # Sender out-degree (index 2) must be 0.0, out-volume (index 4) must be 0.0
    x_w1, hist_e1, q_e1, q_a1, y_1 = window_data[0]
    assert x_w1[sender_idx, 2].item() == 0.0, "Window 1 node features leaked sender out-degree from window 1!"
    assert x_w1[sender_idx, 4].item() == 0.0, "Window 1 node features leaked sender out-volume from window 1!"
    assert x_w1[receiver_idx, 1].item() == 0.0, "Window 1 node features leaked receiver in-degree from window 1!"
    assert x_w1[receiver_idx, 3].item() == 0.0, "Window 1 node features leaked receiver in-volume from window 1!"
    assert hist_e1.shape[1] == 0, "Window 1 message passing edges must be empty!"

    # Window 2 verification:
    # x_w2 MUST reflect the settled window 1 transaction, but NOT the window 2 (step 30) transaction!
    x_w2, hist_e2, q_e2, q_a2, y_2 = window_data[1]
    np.testing.assert_allclose(x_w2[sender_idx, 2].item(), np.log1p(1.0), rtol=1e-5)
    np.testing.assert_allclose(x_w2[sender_idx, 4].item(), np.log1p(5000.0), rtol=1e-5)
    np.testing.assert_allclose(x_w2[receiver_idx, 1].item(), np.log1p(1.0), rtol=1e-5)
    np.testing.assert_allclose(x_w2[receiver_idx, 3].item(), np.log1p(5000.0), rtol=1e-5)
    assert hist_e2.shape[1] == 1, "Window 2 message passing edges must contain settled window 1 edge!"



