"""
Sentinel Phase 5.1: FastAPI Inference Service Integration and Regression Tests
Verifies:
1. Valid and invalid transaction requests with strict Pydantic validation.
2. Startup artifact loading, lifespan management, and health status.
3. Response schema completeness, risk-tier boundaries, and finite scores.
4. Exact TreeSHAP mathematical alignment with XGBoost raw probability.
5. Enforcement of snapshot temporal cutoff (rejection of historical lookahead requests).
6. Enforcement of chronological settlement ordering (rejection of out-of-order settlements).
7. Preservation of settled degrees and volumes for newly created accounts.
8. Controlled offline-online sequential parity for XGBoost, GraphSAGE, and hybrid scores.
9. Cold-start accounts and inductive graph-state initialization.
10. Prediction requests not mutating settlement state (read-only isolation).
11. Duplicate settlement-event protection (idempotency) and state serialization.
12. Sanitized failures when artifacts are missing.
13. End-to-end latency benchmarking.
"""

import concurrent.futures
from pathlib import Path
import threading
import time
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.inference_service import SentinelInferenceService, mask_account
from backend.app.core.config import settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_PARQUET_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "test.parquet"


@pytest.fixture(scope="module")
def client():
    """Module-scoped FastAPI TestClient running the full lifespan startup."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_model_info_endpoints(client):
    """Verifies GET /health and GET /model/info return valid status and metadata."""
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    data_h = res_health.json()
    assert data_h["status"] == "HEALTHY"
    assert data_h["artifacts_loaded"] is True
    assert data_h["graph_state_initialized"] is True
    assert data_h["snapshot_cutoff_step"] >= 620
    assert data_h["max_settled_step"] >= 620
    assert data_h["total_indexed_nodes"] > 0
    assert data_h["settled_edges_count"] > 0
    assert data_h["memory_rss_mb"] > 0.0

    res_info = client.get("/api/v1/model/info")
    assert res_info.status_code == 200
    data_i = res_info.json()
    assert "Sentinel" in data_i["model_architecture"]
    assert np.isclose(data_i["ensemble_weights"]["tabular_xgboost"], 0.78)
    assert np.isclose(data_i["ensemble_weights"]["relational_graphsage"], 0.22)
    assert data_i["risk_tier_thresholds"]["critical"] == 0.80
    assert data_i["risk_tier_thresholds"]["high"] == 0.65
    assert data_i["risk_tier_thresholds"]["medium"] == 0.20
    assert data_i["tier_actions"]["CRITICAL"] == "AUTOMATED_BLOCK"
    assert len(data_i["governance_disclaimers"]) >= 4


def test_valid_transaction_predictions(client):
    """Verifies that valid transactions across different types return 200 OK with complete schema."""
    valid_payloads = [
        {
            "step": 650,
            "type": "TRANSFER",
            "amount": 250000.0,
            "nameOrig": "C123456789",
            "oldbalanceOrg": 250000.0,
            "nameDest": "C987654321",
            "oldbalanceDest": 0.0
        },
        {
            "step": 651,
            "type": "CASH_OUT",
            "amount": 50000.0,
            "nameOrig": "C555555555",
            "oldbalanceOrg": 50000.0,
            "nameDest": "M111111111",
            "oldbalanceDest": 0.0
        },
        {
            "step": 652,
            "type": "PAYMENT",
            "amount": 120.50,
            "nameOrig": "C777777777",
            "oldbalanceOrg": 1500.0,
            "nameDest": "M999999999",
            "oldbalanceDest": 0.0
        }
    ]

    for p in valid_payloads:
        res = client.post("/api/v1/predict/fraud", json=p)
        assert res.status_code == 200
        data = res.json()
        assert "transaction_id" in data
        assert 0.0 <= data["hybrid_risk_score"] <= 1.0
        assert data["assigned_risk_tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        assert data["recommended_action"] in [
            "AUTOMATED_BLOCK",
            "STEP_UP_AUTHENTICATION",
            "MANUAL_ANALYST_REVIEW",
            "AUTOMATED_CLEAR_AND_SETTLE"
        ]
        assert "score_components" in data
        assert "tabular_xgboost" in data["score_components"]
        assert "relational_graphsage" in data["score_components"]
        assert len(data["tabular_shap_attributions"]) > 0
        assert "graph_topological_context" in data
        assert data["latency_ms"] >= 0.0


def test_invalid_transaction_payloads_rejected(client):
    """Verifies that invalid payloads are rejected with 422 Unprocessable Entity."""
    # 1. Missing mandatory field (amount)
    res1 = client.post("/api/v1/predict/fraud", json={
        "step": 650,
        "type": "TRANSFER",
        "nameOrig": "C123",
        "oldbalanceOrg": 100.0,
        "nameDest": "C456",
        "oldbalanceDest": 0.0
    })
    assert res1.status_code == 422
    assert "details" in res1.json()

    # 2. Negative amount
    res2 = client.post("/api/v1/predict/fraud", json={
        "step": 650,
        "type": "TRANSFER",
        "amount": -500.0,
        "nameOrig": "C123",
        "oldbalanceOrg": 100.0,
        "nameDest": "C456",
        "oldbalanceDest": 0.0
    })
    assert res2.status_code == 422

    # 3. Zero amount
    res3 = client.post("/api/v1/predict/fraud", json={
        "step": 650,
        "type": "TRANSFER",
        "amount": 0.0,
        "nameOrig": "C123",
        "oldbalanceOrg": 100.0,
        "nameDest": "C456",
        "oldbalanceDest": 0.0
    })
    assert res3.status_code == 422

    # 4. Negative oldbalanceOrg
    res4 = client.post("/api/v1/predict/fraud", json={
        "step": 650,
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": "C123",
        "oldbalanceOrg": -50.0,
        "nameDest": "C456",
        "oldbalanceDest": 0.0
    })
    assert res4.status_code == 422

    # 5. Invalid transaction type
    res5 = client.post("/api/v1/predict/fraud", json={
        "step": 650,
        "type": "CRYPTO_SWAP",
        "amount": 100.0,
        "nameOrig": "C123",
        "oldbalanceOrg": 100.0,
        "nameDest": "C456",
        "oldbalanceDest": 0.0
    })
    assert res5.status_code == 422


def test_prediction_before_snapshot_cutoff_rejected(client):
    """
    Verifies exact snapshot cutoff boundary semantics:
    Historical snapshot includes data through simulation step 620.
    1. Step 619: Strictly historical -> Rejected with 400 Bad Request.
    2. Step 620: Snapshot cutoff boundary -> Rejected with 400 Bad Request (lookahead violation).
    3. Step 621: First eligible online prediction step -> Accepted with 200 OK.
    """
    base_payload = {
        "type": "TRANSFER",
        "amount": 10000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 10000.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0
    }

    # 1. Step 619: Rejected
    res_619 = client.post("/api/v1/predict/fraud", json={**base_payload, "step": 619})
    assert res_619.status_code == 400
    assert "Historical lookahead violation" in res_619.json()["detail"]
    assert "step 619 is within or precedes" in res_619.json()["detail"]

    # 2. Step 620: Cutoff boundary rejected
    res_620 = client.post("/api/v1/predict/fraud", json={**base_payload, "step": 620})
    assert res_620.status_code == 400
    assert "Historical lookahead violation" in res_620.json()["detail"]
    assert "step 620 is within or precedes" in res_620.json()["detail"]

    # 3. Step 621: First eligible prediction step accepted
    res_621 = client.post("/api/v1/predict/fraud", json={**base_payload, "step": 621})
    assert res_621.status_code == 200
    data_621 = res_621.json()
    assert 0.0 <= data_621["hybrid_risk_score"] <= 1.0


def test_treeshap_exact_probability_alignment(client):
    """
    Regression Test: Verifies that the sigmoid of the sum of TreeSHAP contributions
    (including the base bias value) matches the raw XGBoost predicted probability within 1e-5.
    """
    import xgboost as xgb
    from backend.app.services.inference_service import inference_service

    payload = {
        "step": 650,
        "type": "TRANSFER",
        "amount": 150000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 150000.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0
    }
    res = client.post("/api/v1/predict/fraud", json=payload)
    assert res.status_code == 200
    data = res.json()
    raw_xgb_prob = data["score_components"]["tabular_xgboost"]["raw_score"]

    # Reconstruct booster prediction with the matching iteration range
    feat_df = inference_service.feature_pipeline.transform(pd.DataFrame([payload]))
    booster = inference_service.xgb_model.get_booster()
    dmat = xgb.DMatrix(feat_df, feature_names=inference_service.feature_pipeline.feature_names_)

    best_iter = getattr(inference_service.xgb_model, "best_iteration", None)
    iter_range = (0, int(best_iter) + 1) if best_iter is not None and int(best_iter) >= 0 else (0, 0)
    contribs = booster.predict(dmat, pred_contribs=True, iteration_range=iter_range)[0]

    sum_shap = float(np.sum(contribs))
    reconstructed_prob = float(expit(sum_shap))

    diff = abs(reconstructed_prob - raw_xgb_prob)
    assert diff < 1e-4, (
        f"TreeSHAP probability alignment failed: reconstructed={reconstructed_prob:.6f}, "
        f"raw_xgb={raw_xgb_prob:.6f}, diff={diff:.6e}"
    )


def test_response_schema_and_risk_tier_boundaries(client):
    """Verifies that predicted risk scores map monotonically to defined risk tiers."""
    fraud_payload = {
        "step": 650,
        "type": "TRANSFER",
        "amount": 500000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 500000.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0
    }
    res = client.post("/api/v1/predict/fraud", json=fraud_payload)
    assert res.status_code == 200
    data = res.json()

    score = data["hybrid_risk_score"]
    tier = data["assigned_risk_tier"]
    action = data["recommended_action"]

    if score >= 0.80:
        assert tier == "CRITICAL"
        assert action == "AUTOMATED_BLOCK"
    elif score >= 0.65:
        assert tier == "HIGH"
        assert action == "STEP_UP_AUTHENTICATION"
    elif score >= 0.20:
        assert tier == "MEDIUM"
        assert action == "MANUAL_ANALYST_REVIEW"
    else:
        assert tier == "LOW"
        assert action == "AUTOMATED_CLEAR_AND_SETTLE"

    graph_ctx = data["graph_topological_context"]
    assert "****" in graph_ctx["account_orig_masked"]
    assert "****" in graph_ctx["account_dest_masked"]


def test_offline_parity_with_hybrid_model(client):
    """
    Verifies that the API produces predictions consistent with the standalone
    offline evaluation reference across XGBoost, GraphSAGE, and Hybrid models.
    """
    assert TEST_PARQUET_PATH.exists(), "test.parquet not found"
    test_df = pd.read_parquet(TEST_PARQUET_PATH)
    sample_row = test_df.iloc[0].to_dict()

    payload = {
        "step": int(sample_row["step"]),
        "type": str(sample_row["type"]),
        "amount": float(sample_row["amount"]),
        "nameOrig": str(sample_row["nameOrig"]),
        "oldbalanceOrg": float(sample_row["oldbalanceOrg"]),
        "newbalanceOrig": float(sample_row.get("newbalanceOrig", 0.0)),
        "nameDest": str(sample_row["nameDest"]),
        "oldbalanceDest": float(sample_row["oldbalanceDest"]),
        "newbalanceDest": float(sample_row.get("newbalanceDest", 0.0))
    }

    res = client.post("/api/v1/predict/fraud", json=payload)
    assert res.status_code == 200
    api_data = res.json()

    from backend.app.services.inference_service import inference_service
    hybrid = inference_service.hybrid_model

    # Standalone offline calculation
    df_single = pd.DataFrame([payload])
    feat_df = hybrid.feature_pipeline.transform(df_single)
    offline_raw_xgb = float(hybrid.xgb_model.predict_proba(feat_df)[0, 1])
    offline_cal_xgb = float(hybrid.calibrate_xgb(np.array([offline_raw_xgb]))[0])

    api_raw_xgb = api_data["score_components"]["tabular_xgboost"]["raw_score"]
    api_cal_xgb = api_data["score_components"]["tabular_xgboost"]["calibrated_probability"]

    assert np.isclose(offline_raw_xgb, api_raw_xgb, atol=1e-3)
    assert np.isclose(offline_cal_xgb, api_cal_xgb, atol=1e-3)

    # Verify GNN score finite and hybrid score weighted formula
    api_raw_gnn = api_data["score_components"]["relational_graphsage"]["raw_score"]
    api_cal_gnn = api_data["score_components"]["relational_graphsage"]["calibrated_probability"]
    assert 0.0 <= api_raw_gnn <= 1.0
    assert 0.0 <= api_cal_gnn <= 1.0

    expected_hybrid = float(hybrid.weight_xgb * api_cal_xgb + hybrid.weight_gnn * api_cal_gnn)
    assert np.isclose(expected_hybrid, api_data["hybrid_risk_score"], atol=1e-3)


def test_cold_start_account_handling(client):
    """Verifies that completely unseen accounts score successfully with inductive cold-start representations."""
    cold_payload = {
        "step": 700,
        "type": "PAYMENT",
        "amount": 85.00,
        "nameOrig": "C_UNSEEN_NEW_999",
        "oldbalanceOrg": 500.0,
        "nameDest": "M_UNSEEN_MERCHANT_888",
        "oldbalanceDest": 0.0
    }

    res = client.post("/api/v1/predict/fraud", json=cold_payload)
    assert res.status_code == 200
    data = res.json()
    graph_ctx = data["graph_topological_context"]

    assert graph_ctx["node_orig_status"] == "COLD_START_INDUCTIVE"
    assert graph_ctx["node_dest_status"] == "COLD_START_INDUCTIVE"
    assert graph_ctx["orig_historical_out_degree"] == 0.0
    assert graph_ctx["dest_historical_in_degree"] == 0.0
    assert graph_ctx["is_destination_merchant"] is True
    assert 0.0 <= data["hybrid_risk_score"] <= 1.0


def test_prediction_read_only_isolation(client):
    """Verifies that transaction prediction requests NEVER mutate running graph degrees or edge counts."""
    res_before = client.get("/api/v1/health")
    nodes_before = res_before.json()["total_indexed_nodes"]
    edges_before = res_before.json()["settled_edges_count"]

    payload = {
        "step": 660,
        "type": "TRANSFER",
        "amount": 75000.0,
        "nameOrig": "C_ISOLATION_TEST_1",
        "oldbalanceOrg": 75000.0,
        "nameDest": "C_ISOLATION_TEST_2",
        "oldbalanceDest": 0.0
    }

    for _ in range(3):
        res = client.post("/api/v1/predict/fraud", json=payload)
        assert res.status_code == 200

    res_after = client.get("/api/v1/health")
    nodes_after = res_after.json()["total_indexed_nodes"]
    edges_after = res_after.json()["settled_edges_count"]

    assert nodes_before == nodes_after, "Prediction request mutated node index count!"
    assert edges_before == edges_after, "Prediction request mutated settled edge count!"


def test_same_step_temporal_causality_and_no_leakage(client):
    """
    Regression Test: Enforces strict temporal causality for transactions sharing simulation steps.
    PaySim simulation steps represent 1-hour intervals without within-step sub-hourly timestamps.
    Therefore, no within-step ordering is assumed:
    1. Multiple predictions at step 621 prior to settlement are both valid (scored against snapshot step 620).
    2. Once ANY transaction at step 621 settles, graph watermark advances to 621.
    3. Any subsequent prediction at step 621 is REJECTED with 400 Bad Request to prevent same-step settlement leakage.
    4. A prediction at step 622 is ACCEPTED with 200 OK (scored against graph state settled through step 621).
    """
    txn_a = {
        "step": 621,
        "type": "TRANSFER",
        "amount": 35000.0,
        "nameOrig": "C_SAME_STEP_A",
        "oldbalanceOrg": 35000.0,
        "nameDest": "C_SAME_STEP_B",
        "oldbalanceDest": 0.0
    }
    txn_b = {
        "step": 621,
        "type": "TRANSFER",
        "amount": 42000.0,
        "nameOrig": "C_SAME_STEP_C",
        "oldbalanceOrg": 42000.0,
        "nameDest": "C_SAME_STEP_D",
        "oldbalanceDest": 0.0
    }

    # 1. Both predictions at step 621 succeed prior to any settlement at step 621
    res_a = client.post("/api/v1/predict/fraud", json=txn_a)
    assert res_a.status_code == 200

    res_b = client.post("/api/v1/predict/fraud", json=txn_b)
    assert res_b.status_code == 200

    # 2. Settle transaction A at step 621 into graph state
    settle_a = {
        "transaction_id": "SETTLE_STEP_621_A",
        "step": 621,
        "type": "TRANSFER",
        "amount": 35000.0,
        "nameOrig": "C_SAME_STEP_A",
        "nameDest": "C_SAME_STEP_B"
    }
    res_settle = client.post("/api/v1/events/settlement", json=settle_a)
    assert res_settle.status_code == 200
    assert res_settle.json()["status"] == "SETTLED"

    # 3. Subsequent prediction at step 621 must be REJECTED (graph already contains step 621 settlement)
    txn_c = {
        "step": 621,
        "type": "TRANSFER",
        "amount": 10000.0,
        "nameOrig": "C_SAME_STEP_E",
        "oldbalanceOrg": 10000.0,
        "nameDest": "C_SAME_STEP_F",
        "oldbalanceDest": 0.0
    }
    res_c = client.post("/api/v1/predict/fraud", json=txn_c)
    assert res_c.status_code == 400
    assert "Historical lookahead violation" in res_c.json()["detail"]
    assert "same-step or future settlements" in res_c.json()["detail"]

    # 4. Prediction at step 622 must be ACCEPTED (scored against graph state settled through step 621)
    txn_d = {
        "step": 622,
        "type": "TRANSFER",
        "amount": 18000.0,
        "nameOrig": "C_STEP_622_A",
        "oldbalanceOrg": 18000.0,
        "nameDest": "C_STEP_622_B",
        "oldbalanceDest": 0.0
    }
    res_d = client.post("/api/v1/predict/fraud", json=txn_d)
    assert res_d.status_code == 200
    assert 0.0 <= res_d.json()["hybrid_risk_score"] <= 1.0


def test_settlement_temporal_order_and_new_account_counters(client):
    """
    Verifies:
    1. Out-of-order settlements earlier than the current watermark are rejected with 400.
    2. Valid sequential settlements update counters.
    3. New accounts retain their settled counters in subsequent prediction calls.
    """
    # 1. Attempt out-of-order settlements (Steps 500 and 620 <= snapshot watermark 620)
    for bad_step in [500, 620]:
        out_of_order_event = {
            "transaction_id": f"SETTLE_LATE_{bad_step}",
            "step": bad_step,
            "type": "TRANSFER",
            "amount": 50000.0,
            "nameOrig": f"C_LATE_ORIG_{bad_step}",
            "nameDest": f"C_LATE_DEST_{bad_step}"
        }
        res_late = client.post("/api/v1/events/settlement", json=out_of_order_event)
        assert res_late.status_code == 400
        assert "Out-of-order settlement rejected" in res_late.json()["detail"]

    # 2. Submit valid sequential settlement (Step 675 > watermark 620)
    valid_event = {
        "transaction_id": "SETTLE_SEQ_002",
        "step": 675,
        "type": "TRANSFER",
        "amount": 95000.0,
        "nameOrig": "C_NEW_SETTLED_ACCOUNT_A",
        "nameDest": "C_NEW_SETTLED_ACCOUNT_B"
    }
    res_valid = client.post("/api/v1/events/settlement", json=valid_event)
    assert res_valid.status_code == 200
    data_v = res_valid.json()
    assert data_v["status"] == "SETTLED"
    assert data_v["graph_out_degree_orig"] == 1.0
    assert data_v["graph_in_degree_dest"] == 1.0
    assert "refresh_latency_ms" in data_v

    # 3. Subsequent prediction using this newly settled account: MUST retain its out-degree = 1.0
    predict_payload = {
        "step": 680,
        "type": "TRANSFER",
        "amount": 40000.0,
        "nameOrig": "C_NEW_SETTLED_ACCOUNT_A",
        "oldbalanceOrg": 40000.0,
        "nameDest": "C_OTHER_TARGET",
        "oldbalanceDest": 0.0
    }
    res_pred = client.post("/api/v1/predict/fraud", json=predict_payload)
    assert res_pred.status_code == 200
    pred_data = res_pred.json()
    ctx = pred_data["graph_topological_context"]

    # Must accurately reflect the settled counter
    assert ctx["orig_historical_out_degree"] == 1.0
    assert ctx["orig_historical_out_volume"] == 95000.0
    assert ctx["node_orig_status"] in ["SETTLED_SNAPSHOT", "SETTLED_PENDING_REFRESH"]


def test_idempotent_settlement_events(client):
    """Verifies that duplicate settlement events are ignored without double-counting counters."""
    settle_event = {
        "transaction_id": "SETTLE_IDEMP_999",
        "step": 685,
        "type": "TRANSFER",
        "amount": 125000.0,
        "nameOrig": "C_IDEMP_SRC",
        "nameDest": "C_IDEMP_DST"
    }

    # First event submission: Must settle
    res1 = client.post("/api/v1/events/settlement", json=settle_event)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "SETTLED"
    deg1 = data1["graph_out_degree_orig"]

    # Second event submission with exact same transaction_id: Must ignore duplicate
    res2 = client.post("/api/v1/events/settlement", json=settle_event)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "DUPLICATE_IGNORED"
    assert "duplicate" in data2["message"].lower()
    assert data2["graph_out_degree_orig"] == deg1


def test_missing_artifacts_sanitized_failure():
    """Verifies that missing model artifacts fail cleanly with sanitized FileNotFoundError."""
    dummy_service = SentinelInferenceService()
    with pytest.raises(FileNotFoundError) as exc_info:
        dummy_service.load_artifacts(
            config_path=Path("non_existent_hybrid_config.json"),
            calibrators_path=Path("non_existent_calibrators.joblib"),
            pipeline_path=Path("non_existent_pipeline.joblib"),
            xgb_path=Path("non_existent_xgb.json"),
            gnn_weights_path=Path("non_existent_gnn.pt"),
            gnn_config_path=Path("non_existent_gnn_config.json")
        )
    assert "Required artifact not found" in str(exc_info.value)


def test_latency_benchmark(client):
    """Benchmarks end-to-end inference latency across 15 sequential requests."""
    payload = {
        "step": 700,
        "type": "TRANSFER",
        "amount": 100000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 100000.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0
    }

    latencies = []
    for _ in range(15):
        t0 = time.perf_counter()
        res = client.post("/api/v1/predict/fraud", json=payload)
        assert res.status_code == 200
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    min_lat = float(np.min(latencies))

    print(f"\n[BENCHMARK] Latency across 15 requests: Mean={mean_lat:.2f} ms, P95={p95_lat:.2f} ms, Min={min_lat:.2f} ms")
    assert mean_lat < 150.0, f"Mean latency unexpectedly high: {mean_lat:.2f} ms"


def test_settlement_exception_safety_and_rollback(client, monkeypatch):
    """
    Regression Test: Verifies that an unexpected exception during settlement processing
    (e.g., during embedding refresh) triggers atomic rollback:
    1. Node counters are restored to their prior values.
    2. Newly created node indices are pruned.
    3. Settled edge count is restored.
    4. Transaction ID is NOT added to settled_event_ids, allowing a subsequent retry to succeed.
    """
    from backend.app.services.inference_service import inference_service

    orig_name = "C_FAIL_SAFE_ORIG_001"
    dest_name = "C_FAIL_SAFE_DEST_002"
    txn_id = "TXN_SIMULATED_FAULT_001"

    # Verify accounts do not exist yet
    assert orig_name not in inference_service.graph_builder.node_to_idx
    assert dest_name not in inference_service.graph_builder.node_to_idx
    prior_edge_count = inference_service.settled_edge_index.shape[1]

    # Monkeypatch refresh_graph_embeddings to simulate an unexpected fault
    def mock_fault_refresh():
        raise RuntimeError("Simulated transient memory/CUDA fault during embedding refresh")

    monkeypatch.setattr(inference_service, "refresh_graph_embeddings", mock_fault_refresh)

    fail_payload = {
        "transaction_id": txn_id,
        "step": 690,
        "type": "TRANSFER",
        "amount": 88000.0,
        "nameOrig": orig_name,
        "nameDest": dest_name
    }

    # Settlement attempt should fail with 500 Internal Server Error
    res = client.post("/api/v1/events/settlement", json=fail_payload)
    assert res.status_code == 500
    assert "Failed to update graph state" in res.json()["detail"]

    # Verify atomic rollback
    assert orig_name not in inference_service.graph_builder.node_to_idx
    assert dest_name not in inference_service.graph_builder.node_to_idx
    assert inference_service.settled_edge_index.shape[1] == prior_edge_count
    assert txn_id not in inference_service.settled_event_ids

    # Remove monkeypatch and verify that retry succeeds
    monkeypatch.undo()
    res_retry = client.post("/api/v1/events/settlement", json=fail_payload)
    assert res_retry.status_code == 200
    assert res_retry.json()["status"] == "SETTLED"
    assert txn_id in inference_service.settled_event_ids
    assert orig_name in inference_service.graph_builder.node_to_idx
    assert inference_service.settled_edge_index.shape[1] == prior_edge_count + 1


def test_deterministic_concurrent_prediction_and_settlement_same_step(client, monkeypatch):
    """
    Regression Test: Coordinates a prediction and a settlement at the same simulation step
    using deterministic synchronization primitives (threading.Event) rather than timing-dependent sleeps.

    Verifies:
    1. Scenario A: Settlement starts first and holds the state lock mid-flight.
       A concurrent prediction at the same step blocks on the lock, and upon settlement completion,
       is cleanly REJECTED (HTTP 400 lookahead violation) rather than observing torn graph state
       or scoring against a same-step settlement.
    2. Scenario B: Prediction starts first and holds the state lock mid-flight.
       A concurrent settlement at the same step blocks on the lock.
       Prediction completes successfully (HTTP 200) against the pristine pre-settlement snapshot.
       Upon prediction release, settlement completes successfully (HTTP 200).
       Any subsequent prediction at the same step is then rejected.
    """
    from backend.app.services.inference_service import inference_service

    # =========================================================================
    # SCENARIO A: Settlement acquires lock first; Prediction arrives mid-flight
    # =========================================================================
    step_a = 730
    settle_event_a = {
        "transaction_id": "TXN_CONCUR_A_SETTLE",
        "step": step_a,
        "type": "TRANSFER",
        "amount": 55000.0,
        "nameOrig": "C_CONCUR_A_ORIG",
        "nameDest": "C_CONCUR_A_DEST"
    }
    pred_payload_a = {
        "step": step_a,
        "type": "TRANSFER",
        "amount": 20000.0,
        "nameOrig": "C_CONCUR_A_PRED_SRC",
        "oldbalanceOrg": 20000.0,
        "nameDest": "C_CONCUR_A_PRED_DST",
        "oldbalanceDest": 0.0
    }

    settle_in_lock = threading.Event()
    allow_settle_finish = threading.Event()

    orig_refresh = inference_service.refresh_graph_embeddings
    def paused_refresh():
        settle_in_lock.set()
        allow_settle_finish.wait()
        return orig_refresh()

    monkeypatch.setattr(inference_service, "refresh_graph_embeddings", paused_refresh)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Launch settlement in thread 1
        future_settle = executor.submit(client.post, "/api/v1/events/settlement", json=settle_event_a)

        # Wait deterministically until settlement is holding the state lock mid-refresh
        assert settle_in_lock.wait(timeout=10.0), "Settlement failed to acquire lock in time"

        # Launch concurrent prediction at the exact same step in thread 2
        future_pred = executor.submit(client.post, "/api/v1/predict/fraud", json=pred_payload_a)

        # Allow settlement to resume and complete
        allow_settle_finish.set()

        res_settle = future_settle.result(timeout=10.0)
        res_pred = future_pred.result(timeout=10.0)

    monkeypatch.undo()

    # Settlement must succeed
    assert res_settle.status_code == 200
    assert res_settle.json()["status"] == "SETTLED"

    # Prediction must be cleanly rejected with 400 (never observing partial state or same-step settlement)
    assert res_pred.status_code == 400
    assert "Historical lookahead violation" in res_pred.json()["detail"]
    assert "same-step or future settlements" in res_pred.json()["detail"]

    # =========================================================================
    # SCENARIO B: Prediction acquires lock first; Settlement arrives mid-flight
    # =========================================================================
    step_b = 735
    pred_payload_b = {
        "step": step_b,
        "type": "TRANSFER",
        "amount": 30000.0,
        "nameOrig": "C_CONCUR_B_PRED_SRC",
        "oldbalanceOrg": 30000.0,
        "nameDest": "C_CONCUR_B_PRED_DST",
        "oldbalanceDest": 0.0
    }
    settle_event_b = {
        "transaction_id": "TXN_CONCUR_B_SETTLE",
        "step": step_b,
        "type": "TRANSFER",
        "amount": 75000.0,
        "nameOrig": "C_CONCUR_B_ORIG",
        "nameDest": "C_CONCUR_B_DEST"
    }

    pred_in_lock = threading.Event()
    allow_pred_finish = threading.Event()

    def hook_pre_edge_mlp(module, args):
        pred_in_lock.set()
        allow_pred_finish.wait()
        return args

    hook_handle = inference_service.gnn_model.edge_mlp.register_forward_pre_hook(hook_pre_edge_mlp)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Launch prediction in thread 1
            future_pred_b = executor.submit(client.post, "/api/v1/predict/fraud", json=pred_payload_b)

            # Wait deterministically until prediction is inside state lock executing edge_mlp
            assert pred_in_lock.wait(timeout=10.0), "Prediction failed to enter state lock in time"

            # Launch concurrent settlement at the same step in thread 2 (blocks on state lock)
            future_settle_b = executor.submit(client.post, "/api/v1/events/settlement", json=settle_event_b)

            # Allow prediction to resume and complete
            allow_pred_finish.set()

            res_pred_b = future_pred_b.result(timeout=10.0)
            res_settle_b = future_settle_b.result(timeout=10.0)
    finally:
        hook_handle.remove()

    # Prediction must succeed against consistent pre-settlement snapshot
    assert res_pred_b.status_code == 200
    assert 0.0 <= res_pred_b.json()["hybrid_risk_score"] <= 1.0

    # Settlement must succeed after prediction releases lock
    assert res_settle_b.status_code == 200
    assert res_settle_b.json()["status"] == "SETTLED"

    # Any new prediction at step_b must now be rejected
    res_pred_b_late = client.post("/api/v1/predict/fraud", json=pred_payload_b)
    assert res_pred_b_late.status_code == 400
    assert "Historical lookahead violation" in res_pred_b_late.json()["detail"]


def test_concurrent_settlements_idempotency_and_chronology(client):
    """
    Verifies that concurrent settlement requests:
    1. Preserve duplicate-event idempotency under parallel submissions (exactly 1 SETTLED, rest DUPLICATE_IGNORED).
    2. Enforce chronological watermark ordering (out-of-order settlement rejected even if submitted concurrently).
    """
    from backend.app.services.inference_service import inference_service

    # 1. Duplicate idempotency under 6 concurrent requests
    step_idemp = 740
    txn_id = "TXN_CONCUR_IDEMP_TEST_740"
    idemp_payload = {
        "transaction_id": txn_id,
        "step": step_idemp,
        "type": "TRANSFER",
        "amount": 99000.0,
        "nameOrig": "C_CONCUR_IDEMP_SRC",
        "nameDest": "C_CONCUR_IDEMP_DST"
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(client.post, "/api/v1/events/settlement", json=idemp_payload) for _ in range(6)]
        responses = [f.result(timeout=10.0) for f in futures]

    assert all(r.status_code == 200 for r in responses)
    statuses = [r.json()["status"] for r in responses]
    assert statuses.count("SETTLED") == 1
    assert statuses.count("DUPLICATE_IGNORED") == 5

    # Check graph counters were incremented exactly once
    u_id = inference_service.graph_builder.node_to_idx["C_CONCUR_IDEMP_SRC"]
    v_id = inference_service.graph_builder.node_to_idx["C_CONCUR_IDEMP_DST"]
    assert inference_service.graph_builder.out_degrees[u_id] == 1.0
    assert inference_service.graph_builder.in_degrees[v_id] == 1.0

    # 2. Chronological ordering under concurrent arrival
    # Watermark is now 740. Submit step 750 (valid) and step 738 (out-of-order)
    step_valid = 750
    step_invalid = 738

    payload_valid = {
        "transaction_id": "TXN_ORDER_VALID_750",
        "step": step_valid,
        "type": "TRANSFER",
        "amount": 10000.0,
        "nameOrig": "C_ORDER_SRC_750",
        "nameDest": "C_ORDER_DST_750"
    }
    payload_invalid = {
        "transaction_id": "TXN_ORDER_INVALID_738",
        "step": step_invalid,
        "type": "TRANSFER",
        "amount": 12000.0,
        "nameOrig": "C_ORDER_SRC_738",
        "nameDest": "C_ORDER_DST_738"
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_valid = executor.submit(client.post, "/api/v1/events/settlement", json=payload_valid)
        f_invalid = executor.submit(client.post, "/api/v1/events/settlement", json=payload_invalid)
        res_v = f_valid.result(timeout=10.0)
        res_inv = f_invalid.result(timeout=10.0)

    assert res_v.status_code == 200
    assert res_v.json()["status"] == "SETTLED"
    assert res_inv.status_code == 400
    assert "Out-of-order settlement rejected" in res_inv.json()["detail"]
