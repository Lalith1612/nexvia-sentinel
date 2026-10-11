"""
Sentinel Fraud Inference Service
Singleton inference engine managing model artifacts, stateful graph snapshot,
pre-authorization tabular scoring, relational GraphSAGE evaluation,
Platt probability calibration, TreeSHAP attributions, and idempotent settlement events.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import numpy as np
import pandas as pd
import psutil
from scipy.special import logit
import joblib
import torch
import xgboost as xgb

from ml.features.feature_pipeline import SentinelFeaturePipeline
from ml.features.graph_builder import (
    SentinelGraphBuilder,
    NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    TRANSACTION_TYPES
)
from ml.models.gnn_model import SentinelGraphSAGE
from ml.models.hybrid_model import SentinelHybridRiskModel
from backend.app.core.config import settings
from backend.app.core.database import (
    init_db,
    record_settlement,
    get_settlement,
    get_durable_settlements_since,
    count_settlements
)
from backend.app.schemas.transaction import (
    TransactionRequest,
    SettlementEventRequest,
    SettlementEventResponse,
    FraudPredictionResponse,
    ScoreComponent,
    FeatureAttribution,
    GraphTopologicalContext,
    HealthResponse,
    ModelInfoResponse
)

logger = logging.getLogger("sentinel.inference")


def mask_account(acc: str) -> str:
    """Masks account identifier for privacy-preserving logs and response contexts."""
    if not acc or len(acc) < 4:
        return "ACC***"
    return f"{acc[:3]}****{acc[-2:]}"


class SentinelInferenceService:
    """
    Central inference and graph state management service for Sentinel FastAPI runtime.
    Guarantees:
    1. Single application startup artifact loading (zero retraining/fitting during requests).
    2. Read-only isolation for transaction prediction requests.
    3. Thread-safe, serialized, idempotent processing for settlement events.
    4. Inductive cold-start handling for unseen accounts.
    5. Pure inference mode for PyTorch with eval() state.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self.feature_pipeline: Optional[SentinelFeaturePipeline] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.gnn_model: Optional[SentinelGraphSAGE] = None
        self.hybrid_model: Optional[SentinelHybridRiskModel] = None
        self.graph_builder: Optional[SentinelGraphBuilder] = None

        # Graph runtime state
        self.settled_edge_index: Optional[torch.Tensor] = None
        self.cached_node_embeddings: Optional[torch.Tensor] = None
        self.cold_start_customer_emb: Optional[torch.Tensor] = None
        self.cold_start_merchant_emb: Optional[torch.Tensor] = None

        # Idempotency and settlement tracking
        self.settled_event_ids: Set[str] = set()
        self.db_path: Path = settings.LEDGER_DB_PATH
        self._lock = asyncio.Lock()
        self._state_lock = threading.RLock()
        self.snapshot_cutoff_step: int = 0
        self.max_settled_step: int = 0
        self.is_initialized: bool = False
        self.initialization_error: Optional[str] = None

    def load_artifacts(
        self,
        config_path: Path = settings.HYBRID_CONFIG_PATH,
        calibrators_path: Path = settings.CALIBRATORS_PATH,
        pipeline_path: Path = settings.FEATURE_PIPELINE_PATH,
        xgb_path: Path = settings.XGB_MODEL_PATH,
        gnn_weights_path: Path = settings.GNN_WEIGHTS_PATH,
        gnn_config_path: Path = settings.GNN_CONFIG_PATH
    ):
        """Loads trained models, pipelines, calibrators, and architectures into memory."""
        logger.info("Loading Sentinel model artifacts from disk...")
        for p in [config_path, calibrators_path, pipeline_path, xgb_path, gnn_weights_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required artifact not found: {p}")

        # 1. Feature Pipeline
        self.feature_pipeline = joblib.load(pipeline_path)

        # 2. XGBoost Baseline
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(xgb_path)

        # 3. GraphSAGE Architecture & Weights
        with open(gnn_config_path, "r") as f:
            gnn_cfg = json.load(f)
        hp = gnn_cfg.get("hyperparameters", {})
        hidden_channels = hp.get("hidden_channels", 64)
        out_channels = hp.get("out_channels", 32)
        dropout = hp.get("dropout", 0.2)

        self.gnn_model = SentinelGraphSAGE(
            node_in_channels=len(NODE_FEATURE_NAMES),
            edge_in_channels=len(EDGE_FEATURE_NAMES),
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout
        ).to(self.device)
        self.gnn_model.load_state_dict(torch.load(gnn_weights_path, map_location=self.device))
        self.gnn_model.eval()

        # 4. Hybrid Risk Model & Calibrators
        self.hybrid_model = SentinelHybridRiskModel.load(
            config_path=config_path,
            calibrators_path=calibrators_path,
            pipeline_path=pipeline_path,
            xgb_path=xgb_path
        )
        self.hybrid_model.gnn_model = self.gnn_model

        # 5. Precompute Cold-Start Inductive Embeddings
        with torch.inference_mode():
            x_cust = torch.zeros((1, len(NODE_FEATURE_NAMES)), dtype=torch.float, device=self.device)
            x_merch = torch.zeros((1, len(NODE_FEATURE_NAMES)), dtype=torch.float, device=self.device)
            x_merch[0, 0] = 1.0  # is_merchant = 1.0
            empty_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)

            self.cold_start_customer_emb = self.gnn_model.encode_nodes(x_cust, empty_edges)[0]
            self.cold_start_merchant_emb = self.gnn_model.encode_nodes(x_merch, empty_edges)[0]

        logger.info("Successfully loaded feature pipeline, XGBoost, GraphSAGE, and calibrators.")

    def initialize_graph_state(
        self,
        seed_df: Optional[pd.DataFrame] = None,
        load_baseline_partitions: bool = True
    ):
        """
        Initializes graph builder state and computes baseline node embeddings snapshot.
        If seed_df is given, seeds from seed_df.
        Otherwise, if load_baseline_partitions is True, seeds from train_benchmark_sample + val parquets.
        """
        logger.info("Initializing in-memory graph state snapshot...")
        self.graph_builder = SentinelGraphBuilder()

        if seed_df is not None:
            logger.info("Seeding graph state from custom seed DataFrame (%d rows)...", len(seed_df))
            data, _, _ = self.graph_builder.build_graph_data(seed_df, update_state=True)
            self.settled_edge_index = data.edge_index
            self.snapshot_cutoff_step = int(seed_df["step"].max()) if (len(seed_df) > 0 and "step" in seed_df.columns) else 0
            self.max_settled_step = self.snapshot_cutoff_step
        elif load_baseline_partitions and settings.TRAIN_SAMPLE_PATH.exists() and settings.VAL_PATH.exists():
            logger.info("Seeding historical graph state from baseline processed partitions...")
            train_sample_df = pd.read_parquet(settings.TRAIN_SAMPLE_PATH)
            val_df = pd.read_parquet(settings.VAL_PATH)

            train_data, _, _ = self.graph_builder.build_graph_data(train_sample_df, update_state=True)
            val_data, _, _ = self.graph_builder.build_graph_data(val_df, update_state=True)
            self.settled_edge_index = torch.cat([train_data.edge_index, val_data.edge_index], dim=1)
            self.snapshot_cutoff_step = int(max(train_sample_df["step"].max(), val_df["step"].max()))
            self.max_settled_step = self.snapshot_cutoff_step
        else:
            logger.warning("No baseline dataset available; initializing empty graph state snapshot.")
            self.settled_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.snapshot_cutoff_step = 0
            self.max_settled_step = 0

        # Restart Recovery: Initialize durable ledger schema and replay post-cutoff settlements
        self.init_database()
        self.recover_durable_settlements()

        self.refresh_graph_embeddings()
        self.hybrid_model.graph_builder = self.graph_builder
        self.is_initialized = True
        logger.info(
            "Graph state ready: %d unique accounts, %d settled edges indexed (snapshot cutoff step: %d, max watermark: %d).",
            len(self.graph_builder.idx_to_node),
            self.settled_edge_index.shape[1],
            self.snapshot_cutoff_step,
            self.max_settled_step
        )

    def init_database(self, db_path: Optional[Path] = None):
        """Initializes durable SQLite ledger schema."""
        if db_path is not None:
            self.db_path = db_path
        init_db(self.db_path)

    def reconcile_pending_settlements(
        self,
        exclude_txn_id: Optional[str] = None,
        auto_refresh_embeddings: bool = True
    ) -> int:
        """
        Replays durable post-snapshot settlements from SQLite in strict chronological order
        that have not yet been reflected in in-memory graph state.
        Rebuilds graph state deterministically without lookahead or data loss.
        """
        with self._state_lock:
            durable_events = get_durable_settlements_since(self.db_path, self.snapshot_cutoff_step)
            if not durable_events:
                return 0

            unreconciled = [
                row for row in durable_events
                if row["transaction_id"] not in self.settled_event_ids
                and row["transaction_id"] != exclude_txn_id
            ]
            if not unreconciled:
                return 0

            logger.info(
                "Reconciling %d pending durable settlements from ledger (cutoff step %d)...",
                len(unreconciled),
                self.snapshot_cutoff_step
            )
            for row in unreconciled:
                u = self.graph_builder.get_or_create_node_id(row["name_orig"])
                v = self.graph_builder.get_or_create_node_id(row["name_dest"])
                amt = float(row["amount"])

                prior_u_deg = self.graph_builder.out_degrees.get(u, 0.0)
                prior_u_vol = self.graph_builder.out_volume.get(u, 0.0)
                prior_v_deg = self.graph_builder.in_degrees.get(v, 0.0)
                prior_v_vol = self.graph_builder.in_volume.get(v, 0.0)

                self.graph_builder.out_degrees[u] = prior_u_deg + 1.0
                self.graph_builder.out_volume[u] = prior_u_vol + amt
                self.graph_builder.in_degrees[v] = prior_v_deg + 1.0
                self.graph_builder.in_volume[v] = prior_v_vol + amt

                new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=self.device)
                if self.settled_edge_index is None or self.settled_edge_index.numel() == 0:
                    self.settled_edge_index = new_edge
                else:
                    self.settled_edge_index = torch.cat([self.settled_edge_index, new_edge], dim=1)

                self.max_settled_step = max(self.max_settled_step, row["step"])
                self.settled_event_ids.add(row["transaction_id"])
                logger.info("Reconciled pending settlement %s (step %d) into graph.", row["transaction_id"], row["step"])

            if auto_refresh_embeddings:
                self.refresh_graph_embeddings()

            logger.info(
                "Successfully reconciled %d durable settlements. Active watermark advanced to step %d.",
                len(unreconciled),
                self.max_settled_step
            )
            return len(unreconciled)

    def recover_durable_settlements(self) -> int:
        """
        Replays durable post-snapshot settlements from SQLite in strict chronological order.
        Rebuilds graph state deterministically without lookahead or data loss.
        """
        return self.reconcile_pending_settlements(auto_refresh_embeddings=False)

    def refresh_graph_embeddings(self):
        """Recomputes node embeddings across the current settled graph snapshot."""
        if self.graph_builder is None or self.gnn_model is None:
            return
        with self._state_lock:
            n_nodes = len(self.graph_builder.idx_to_node)
            if n_nodes == 0 or self.settled_edge_index is None:
                self.cached_node_embeddings = torch.empty((0, self.gnn_model.out_channels), device=self.device)
                return

            x_hist = self.graph_builder.get_current_node_features().to(self.device)
            with torch.inference_mode():
                self.cached_node_embeddings = self.gnn_model.encode_nodes(x_hist, self.settled_edge_index.to(self.device))
            logger.debug("Refreshed embeddings snapshot for %d nodes.", n_nodes)

    def predict_transaction(
        self,
        txn: TransactionRequest,
        transaction_id: Optional[str] = None
    ) -> FraudPredictionResponse:
        """
        Executes pre-authorization fraud scoring for a single transaction.
        STRICTLY READ-ONLY: Never updates running graph state, node indices, or settled edges.
        Enforces temporal boundary: rejects transactions before snapshot cutoff.
        """
        start_time = time.perf_counter()
        if not self.is_initialized or self.hybrid_model is None:
            raise RuntimeError("Inference service is not initialized or model artifacts are missing.")

        # Temporal boundary & causal watermark validation
        active_watermark = max(self.snapshot_cutoff_step, self.max_settled_step)
        if active_watermark > 0 and txn.step <= active_watermark:
            raise ValueError(
                f"Historical lookahead violation: Transaction step {txn.step} is within or precedes the initialized "
                f"graph snapshot cutoff / current settled watermark (step {active_watermark}). Transactions cannot "
                f"be scored against graph state containing same-step or future settlements. The first eligible prediction step is {active_watermark + 1}."
            )

        txn_id = transaction_id or str(uuid.uuid4())
        txn_dict = txn.model_dump()

        # 1. Tabular XGBoost Pre-Authorization Scoring
        df_single = pd.DataFrame([txn_dict])
        feat_df = self.feature_pipeline.transform(df_single)
        raw_xgb_prob = float(self.xgb_model.predict_proba(feat_df)[0, 1])
        cal_xgb_prob = float(self.hybrid_model.calibrate_xgb(np.array([raw_xgb_prob]))[0])

        # 2. Native TreeSHAP Explanations with exact iteration_range alignment
        booster = self.xgb_model.get_booster()
        dmat = xgb.DMatrix(feat_df, feature_names=self.feature_pipeline.feature_names_)
        best_iter = getattr(self.xgb_model, "best_iteration", None)
        if best_iter is not None and int(best_iter) >= 0:
            iter_range = (0, int(best_iter) + 1)
        else:
            iter_range = (0, 0)
        contribs = booster.predict(dmat, pred_contribs=True, iteration_range=iter_range)[0]
        feat_contribs = contribs[:-1]
        feature_names = self.feature_pipeline.feature_names_

        top_indices = np.argsort(np.abs(feat_contribs))[::-1][:5]
        shap_explanations: List[FeatureAttribution] = []
        for idx in top_indices:
            fname = feature_names[idx]
            fval = float(feat_df.iloc[0, idx])
            val_contrib = float(feat_contribs[idx])
            shap_explanations.append(
                FeatureAttribution(
                    feature=fname,
                    value=round(fval, 4),
                    shap_contribution=round(val_contrib, 4),
                    impact="INCREASES_FRAUD_RISK" if val_contrib > 0 else "REDUCES_FRAUD_RISK"
                )
            )

        # 3. Relational GraphSAGE Pre-Authorization Scoring (Read-Only)
        orig_name = str(txn.nameOrig)
        dest_name = str(txn.nameDest)
        is_dest_merchant = dest_name.startswith("M")

        with self._state_lock:
            # Re-check watermark under state lock to eliminate race with concurrent settlement
            active_watermark = max(self.snapshot_cutoff_step, self.max_settled_step)
            if active_watermark > 0 and txn.step <= active_watermark:
                raise ValueError(
                    f"Historical lookahead violation: Transaction step {txn.step} is within or precedes the initialized "
                    f"graph snapshot cutoff / current settled watermark (step {active_watermark}). Transactions cannot "
                    f"be scored against graph state containing same-step or future settlements. The first eligible prediction step is {active_watermark + 1}."
                )

            u_id = self.graph_builder.node_to_idx.get(orig_name, None)
            v_id = self.graph_builder.node_to_idx.get(dest_name, None)

            # Retrieve counters directly from authoritative graph state
            if u_id is not None:
                src_prior_out_deg = float(self.graph_builder.out_degrees.get(u_id, 0.0))
                src_prior_out_vol = float(self.graph_builder.out_volume.get(u_id, 0.0))
            else:
                src_prior_out_deg = 0.0
                src_prior_out_vol = 0.0

            if v_id is not None:
                dst_prior_in_deg = float(self.graph_builder.in_degrees.get(v_id, 0.0))
                dst_prior_in_vol = float(self.graph_builder.in_volume.get(v_id, 0.0))
            else:
                dst_prior_in_deg = 0.0
                dst_prior_in_vol = 0.0

            # Structural node embeddings lookup
            if u_id is not None and self.cached_node_embeddings is not None and u_id < len(self.cached_node_embeddings):
                h_u = self.cached_node_embeddings[u_id]
                node_orig_status = "SETTLED_SNAPSHOT"
            elif u_id is not None:
                h_u = self.cold_start_customer_emb
                node_orig_status = "SETTLED_PENDING_REFRESH"
            else:
                h_u = self.cold_start_customer_emb
                node_orig_status = "COLD_START_INDUCTIVE"

            if v_id is not None and self.cached_node_embeddings is not None and v_id < len(self.cached_node_embeddings):
                h_v = self.cached_node_embeddings[v_id]
                node_dest_status = "SETTLED_SNAPSHOT"
            elif v_id is not None:
                h_v = self.cold_start_merchant_emb if is_dest_merchant else self.cold_start_customer_emb
                node_dest_status = "SETTLED_PENDING_REFRESH"
            else:
                h_v = self.cold_start_merchant_emb if is_dest_merchant else self.cold_start_customer_emb
                node_dest_status = "COLD_START_INDUCTIVE"

            # Construct candidate edge attribute vector (15 features, strictly prior counters)
            amt = float(txn.amount)
            old_orig = float(txn.oldbalanceOrg)
            old_dest = float(txn.oldbalanceDest)
            step = int(txn.step)
            hour = float(step % 24)
            day = float((step // 24) % 7)

            amt_log = np.log1p(max(0.0, amt))
            orig_bal_log = np.log1p(max(0.0, old_orig))
            dest_bal_log = np.log1p(max(0.0, old_dest))
            amt_ratio = amt / (old_orig + 1.0)
            type_encs = [1.0 if txn.type == t else 0.0 for t in TRANSACTION_TYPES]

            edge_attr_vec = np.array([
                amt_log,
                orig_bal_log,
                dest_bal_log,
                amt_ratio,
                hour,
                day,
                *type_encs,
                np.log1p(src_prior_out_deg),
                np.log1p(src_prior_out_vol),
                np.log1p(dst_prior_in_deg),
                np.log1p(dst_prior_in_vol)
            ], dtype=np.float32)

            edge_attr_t = torch.tensor(edge_attr_vec, dtype=torch.float, device=self.device)

            # Concatenate edge representation: [h_u (32) || h_v (32) || edge_attr (15)] -> (79)
            edge_repr = torch.cat([h_u, h_v, edge_attr_t], dim=-1).unsqueeze(0)

            with torch.inference_mode():
                logits = self.gnn_model.edge_mlp(edge_repr).squeeze(-1)
                raw_gnn_prob = float(torch.sigmoid(logits).item())

        cal_gnn_prob = float(self.hybrid_model.calibrate_gnn(np.array([raw_gnn_prob]))[0])

        # 4. Hybrid Probability Fusion & Risk Tier Assignment
        hybrid_prob = float(
            self.hybrid_model.weight_xgb * cal_xgb_prob +
            self.hybrid_model.weight_gnn * cal_gnn_prob
        )
        hybrid_prob = max(0.0, min(1.0, hybrid_prob))
        tier = self.hybrid_model.classify_tier(hybrid_prob)
        action = self.hybrid_model.ACTION_MAP[tier]

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        graph_context = GraphTopologicalContext(
            account_orig_masked=mask_account(orig_name),
            account_dest_masked=mask_account(dest_name),
            is_destination_merchant=is_dest_merchant,
            orig_historical_out_degree=src_prior_out_deg,
            orig_historical_out_volume=round(src_prior_out_vol, 2),
            dest_historical_in_degree=dst_prior_in_deg,
            dest_historical_in_volume=round(dst_prior_in_vol, 2),
            raw_gnn_anomaly_score=round(raw_gnn_prob, 4),
            calibrated_gnn_risk=round(cal_gnn_prob, 4),
            relational_weight_in_ensemble=round(self.hybrid_model.weight_gnn, 2),
            node_orig_status=node_orig_status,
            node_dest_status=node_dest_status
        )

        return FraudPredictionResponse(
            transaction_id=txn_id,
            hybrid_risk_score=round(hybrid_prob, 4),
            assigned_risk_tier=tier,
            recommended_action=action,
            score_components={
                "tabular_xgboost": ScoreComponent(
                    raw_score=round(raw_xgb_prob, 4),
                    calibrated_probability=round(cal_xgb_prob, 4),
                    ensemble_weight=round(self.hybrid_model.weight_xgb, 2),
                    weighted_contribution=round(self.hybrid_model.weight_xgb * cal_xgb_prob, 4)
                ),
                "relational_graphsage": ScoreComponent(
                    raw_score=round(raw_gnn_prob, 4),
                    calibrated_probability=round(cal_gnn_prob, 4),
                    ensemble_weight=round(self.hybrid_model.weight_gnn, 2),
                    weighted_contribution=round(self.hybrid_model.weight_gnn * cal_gnn_prob, 4)
                )
            },
            tabular_shap_attributions=shap_explanations,
            graph_topological_context=graph_context,
            regulatory_disclaimers=self.hybrid_model.DISCLAIMERS,
            latency_ms=round(total_latency_ms, 2)
        )

    async def process_settlement_event(
        self,
        event: SettlementEventRequest,
        auto_refresh_embeddings: bool = True
    ) -> SettlementEventResponse:
        """
        Processes confirmed settlement events into graph history.
        Thread-safe, serialized via async lock.
        Idempotent: duplicate transaction_ids return DUPLICATE_IGNORED without modifying graph state.
        Enforces chronological ordering: rejects events with step earlier than max_settled_step.
        """
        async with self._lock:
            with self._state_lock:
                # 1. Idempotency check first
                if event.transaction_id in self.settled_event_ids:
                    logger.info("Duplicate settlement event ignored: %s", event.transaction_id)
                    u_id = self.graph_builder.node_to_idx.get(event.nameOrig, 0)
                    v_id = self.graph_builder.node_to_idx.get(event.nameDest, 0)
                    return SettlementEventResponse(
                        transaction_id=event.transaction_id,
                        status="DUPLICATE_IGNORED",
                        settled_at=datetime.now(timezone.utc).isoformat(),
                        message="Transaction was previously settled into graph state; duplicate event ignored.",
                        graph_out_degree_orig=float(self.graph_builder.out_degrees.get(u_id, 0.0)),
                        graph_in_degree_dest=float(self.graph_builder.in_degrees.get(v_id, 0.0))
                    )

                # 2. Reconcile any prior pending durable settlements in SQLite (excluding current event if it's already in DB)
                self.reconcile_pending_settlements(
                    exclude_txn_id=event.transaction_id,
                    auto_refresh_embeddings=auto_refresh_embeddings
                )

                # Check if already recorded in durable SQLite ledger (recovery case where prior in-memory update failed)
                existing_record = get_settlement(self.db_path, event.transaction_id)
                already_persisted_in_db = existing_record is not None

                # 2. Chronological watermark check
                if (self.snapshot_cutoff_step > 0 and event.step <= self.snapshot_cutoff_step) or (self.max_settled_step > 0 and event.step < self.max_settled_step):
                    cutoff_watermark = max(self.snapshot_cutoff_step, self.max_settled_step)
                    raise ValueError(
                        f"Out-of-order settlement rejected: Event step {event.step} is earlier than or within current "
                        f"graph settlement watermark (step {cutoff_watermark}). Historical graph mutations "
                        f"must be strictly chronological to prevent temporal contamination."
                    )

                # 3. Snapshot prior state for atomic rollback on unexpected exception
                orig_u_existed = event.nameOrig in self.graph_builder.node_to_idx
                orig_v_existed = event.nameDest in self.graph_builder.node_to_idx
                prior_idx_to_node_len = len(self.graph_builder.idx_to_node)
                prior_max_settled_step = self.max_settled_step
                prior_settled_edge_index = self.settled_edge_index

                settled_at = datetime.now(timezone.utc).isoformat()

                # 4. Commit to durable SQLite ledger first (if not already recorded)
                if not already_persisted_in_db:
                    record_settlement(
                        self.db_path,
                        transaction_id=event.transaction_id,
                        step=event.step,
                        type_=event.type,
                        amount=float(event.amount),
                        name_orig=event.nameOrig,
                        name_dest=event.nameDest,
                        settled_at=settled_at
                    )

                try:
                    # 5. Update running graph state
                    u = self.graph_builder.get_or_create_node_id(event.nameOrig)
                    v = self.graph_builder.get_or_create_node_id(event.nameDest)
                    amt = float(event.amount)

                    prior_u_out_deg = self.graph_builder.out_degrees.get(u, 0.0)
                    prior_u_out_vol = self.graph_builder.out_volume.get(u, 0.0)
                    prior_v_in_deg = self.graph_builder.in_degrees.get(v, 0.0)
                    prior_v_in_vol = self.graph_builder.in_volume.get(v, 0.0)

                    self.graph_builder.out_degrees[u] = prior_u_out_deg + 1.0
                    self.graph_builder.out_volume[u] = prior_u_out_vol + amt
                    self.graph_builder.in_degrees[v] = prior_v_in_deg + 1.0
                    self.graph_builder.in_volume[v] = prior_v_in_vol + amt

                    # Append edge to settled_edge_index
                    new_edge = torch.tensor([[u], [v]], dtype=torch.long, device=self.device)
                    if self.settled_edge_index is None or self.settled_edge_index.numel() == 0:
                        self.settled_edge_index = new_edge
                    else:
                        self.settled_edge_index = torch.cat([self.settled_edge_index, new_edge], dim=1)

                    self.max_settled_step = max(self.max_settled_step, event.step)

                    # 5. Correctness-first Embedding Refresh
                    t_ref_start = time.perf_counter()
                    if auto_refresh_embeddings:
                        self.refresh_graph_embeddings()
                        refresh_latency_ms = (time.perf_counter() - t_ref_start) * 1000.0
                    else:
                        refresh_latency_ms = 0.0

                    # 6. Record event for idempotency ONLY upon complete success
                    self.settled_event_ids.add(event.transaction_id)
                except Exception as exc:
                    # Atomic rollback of graph state to prevent partial corruption
                    if "u" in locals():
                        self.graph_builder.out_degrees[u] = prior_u_out_deg
                        self.graph_builder.out_volume[u] = prior_u_out_vol
                    if "v" in locals():
                        self.graph_builder.in_degrees[v] = prior_v_in_deg
                        self.graph_builder.in_volume[v] = prior_v_in_vol
                    if not orig_u_existed and event.nameOrig in self.graph_builder.node_to_idx:
                        del self.graph_builder.node_to_idx[event.nameOrig]
                        self.graph_builder.out_degrees.pop(u, None)
                        self.graph_builder.out_volume.pop(u, None)
                    if not orig_v_existed and event.nameDest in self.graph_builder.node_to_idx:
                        del self.graph_builder.node_to_idx[event.nameDest]
                        self.graph_builder.in_degrees.pop(v, None)
                        self.graph_builder.in_volume.pop(v, None)
                    del self.graph_builder.idx_to_node[prior_idx_to_node_len:]
                    self.settled_edge_index = prior_settled_edge_index
                    self.max_settled_step = prior_max_settled_step
                    logger.error("Exception during settlement event processing; rolled back state: %s", str(exc))
                    raise

                logger.info("Settled transaction %s into graph state (step %d).", event.transaction_id, event.step)

                return SettlementEventResponse(
                    transaction_id=event.transaction_id,
                    status="SETTLED",
                    settled_at=datetime.now(timezone.utc).isoformat(),
                    message="Transaction successfully settled into graph state ledger.",
                    graph_out_degree_orig=float(self.graph_builder.out_degrees[u]),
                    graph_in_degree_dest=float(self.graph_builder.in_degrees[v]),
                    refresh_latency_ms=round(refresh_latency_ms, 2)
                )

    def get_health_status(self) -> HealthResponse:
        """Returns health diagnostics and memory footprint."""
        process = psutil.Process()
        mem_rss_mb = process.memory_info().rss / (1024 * 1024)

        artifacts_loaded = bool(
            self.feature_pipeline is not None and
            self.xgb_model is not None and
            self.gnn_model is not None and
            self.hybrid_model is not None
        )
        graph_ready = bool(
            self.graph_builder is not None and
            self.is_initialized
        )

        status = "HEALTHY" if (artifacts_loaded and graph_ready) else "DEGRADED"
        if not artifacts_loaded:
            status = "UNHEALTHY"

        total_nodes = len(self.graph_builder.idx_to_node) if self.graph_builder else 0
        total_edges = self.settled_edge_index.shape[1] if self.settled_edge_index is not None else 0

        return HealthResponse(
            status=status,
            service_name=settings.PROJECT_NAME,
            version=settings.VERSION,
            artifacts_loaded=artifacts_loaded,
            graph_state_initialized=graph_ready,
            snapshot_cutoff_step=self.snapshot_cutoff_step,
            max_settled_step=self.max_settled_step,
            total_indexed_nodes=total_nodes,
            settled_edges_count=total_edges,
            torch_device=str(self.device),
            memory_rss_mb=round(mem_rss_mb, 2)
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Returns architectural parameters, validation metadata, and regulatory disclaimers."""
        if self.hybrid_model is None:
            raise RuntimeError("Hybrid model is not loaded.")

        with open(settings.HYBRID_CONFIG_PATH, "r") as f:
            cfg = json.load(f)

        return ModelInfoResponse(
            model_architecture=cfg.get("model_architecture", "Sentinel Hybrid Ensemble (XGBoost + GraphSAGE)"),
            ensemble_weights=cfg.get("ensemble_weights", {
                "tabular_xgboost": self.hybrid_model.weight_xgb,
                "relational_graphsage": self.hybrid_model.weight_gnn
            }),
            risk_tier_thresholds=cfg.get("risk_tier_thresholds", {
                "critical": self.hybrid_model.threshold_critical,
                "high": self.hybrid_model.threshold_high,
                "medium": self.hybrid_model.threshold_medium,
                "low": self.hybrid_model.threshold_low
            }),
            tier_actions=cfg.get("tier_actions", self.hybrid_model.ACTION_MAP),
            calibration_method=cfg.get("calibration", {}).get(
                "method", "Platt Scaling (Univariate Logistic Regression on Log-Odds)"
            ),
            validation_dataset=cfg.get("validation_dataset", "val.parquet (Steps 501-620)"),
            governance_disclaimers=self.hybrid_model.DISCLAIMERS,
            operational_prototype_boundary=(
                "Sentinel is an architectural research prototype evaluated on synthetic PaySim data. "
                "The 4 risk tiers and recommended actions are illustrative decision-layer demonstrations, "
                "not validated real-world banking policies. The API does not execute or block actual financial transfers."
            )
        )


inference_service = SentinelInferenceService()
