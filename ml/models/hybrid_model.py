"""
Sentinel Hybrid Risk Combination Model & Explainability Engine
Combines Tabular Gradient Boosted Trees (XGBoost) and Relational Graph Neural Network (GraphSAGE).
Features:
- Dual-model calibrated probability fusion (Platt scaling)
- 4-Tier Operational Risk Classification (Critical, High, Medium, Low)
- Dual-domain Explainability: Native TreeSHAP + Relational Graph Topological Attribution
- Explicit Governance Warnings & Methodological Limitation Disclosures
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.special import logit
import joblib
import torch
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "ml" / "models"


class SentinelHybridRiskModel:
    """
    Production-style wrapper combining XGBoost and GraphSAGE with validation-calibrated
    probability fusion and multi-tier explainability.
    """

    TIER_CRITICAL = "CRITICAL"
    TIER_HIGH = "HIGH"
    TIER_MEDIUM = "MEDIUM"
    TIER_LOW = "LOW"

    ACTION_MAP = {
        TIER_CRITICAL: "AUTOMATED_BLOCK",
        TIER_HIGH: "STEP_UP_AUTHENTICATION",
        TIER_MEDIUM: "MANUAL_ANALYST_REVIEW",
        TIER_LOW: "AUTOMATED_CLEAR_AND_SETTLE"
    }

    DISCLAIMERS = [
        "PaySim Simulation Topology: Transactions are generated as synthetic disjoint pairs; genuine multi-hop money laundering chains and complex cyclic fraud rings are absent in this dataset.",
        "Temporal Batching Resolution: Graph neural network embeddings reflect 25-hour causal training windows evaluated against hourly streaming test increments.",
        "Attribution vs. Causal Intent: Feature and graph attributions represent statistical model sensitivities, not forensic or legal proof of fraud.",
        "Research Prototype Status: Developed for research and educational purposes; not certified or hardened for live core-banking financial systems."
    ]

    def __init__(
        self,
        weight_xgb: float = 0.80,
        weight_gnn: float = 0.20,
        threshold_critical: float = 0.80,
        threshold_high: float = 0.65,
        threshold_medium: float = 0.20,
        calibrator_xgb=None,
        calibrator_gnn=None,
        feature_pipeline=None,
        xgb_model=None,
        gnn_model=None,
        graph_builder=None
    ):
        self.weight_xgb = float(weight_xgb)
        self.weight_gnn = float(weight_gnn)
        self.threshold_critical = float(threshold_critical)
        self.threshold_high = float(threshold_high)
        self.threshold_medium = float(threshold_medium)
        self.threshold_low = 0.0

        self.calibrator_xgb = calibrator_xgb
        self.calibrator_gnn = calibrator_gnn
        self.feature_pipeline = feature_pipeline
        self.xgb_model = xgb_model
        self.gnn_model = gnn_model
        self.graph_builder = graph_builder

    def calibrate_xgb(self, raw_probs: np.ndarray) -> np.ndarray:
        """Applies validation-fitted Platt scaling to raw XGBoost probabilities."""
        if self.calibrator_xgb is None:
            return raw_probs
        clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        log_odds = logit(clipped).reshape(-1, 1)
        return self.calibrator_xgb.predict_proba(log_odds)[:, 1]

    def calibrate_gnn(self, raw_probs: np.ndarray) -> np.ndarray:
        """Applies validation-fitted Platt scaling to raw GNN probabilities."""
        if self.calibrator_gnn is None:
            return raw_probs
        clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        log_odds = logit(clipped).reshape(-1, 1)
        return self.calibrator_gnn.predict_proba(log_odds)[:, 1]

    def combine_probabilities(
        self,
        raw_xgb_probs: np.ndarray,
        raw_gnn_probs: np.ndarray,
        calibrate: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes calibrated hybrid risk probabilities.
        Returns: (hybrid_probs, cal_xgb_probs, cal_gnn_probs)
        """
        if calibrate:
            cal_xgb = self.calibrate_xgb(raw_xgb_probs)
            cal_gnn = self.calibrate_gnn(raw_gnn_probs)
        else:
            cal_xgb = raw_xgb_probs
            cal_gnn = raw_gnn_probs

        hybrid = self.weight_xgb * cal_xgb + self.weight_gnn * cal_gnn
        return hybrid, cal_xgb, cal_gnn

    def classify_tier(self, score: float) -> str:
        """Assigns a transaction risk score to a mutually exclusive risk tier."""
        if score >= self.threshold_critical:
            return self.TIER_CRITICAL
        elif score >= self.threshold_high:
            return self.TIER_HIGH
        elif score >= self.threshold_medium:
            return self.TIER_MEDIUM
        else:
            return self.TIER_LOW

    def classify_tiers(self, scores: np.ndarray) -> np.ndarray:
        """Vectorized risk tier classification."""
        tiers = np.empty(len(scores), dtype=object)
        tiers[scores >= self.threshold_critical] = self.TIER_CRITICAL
        tiers[(scores >= self.threshold_high) & (scores < self.threshold_critical)] = self.TIER_HIGH
        tiers[(scores >= self.threshold_medium) & (scores < self.threshold_high)] = self.TIER_MEDIUM
        tiers[scores < self.threshold_medium] = self.TIER_LOW
        return tiers

    def explain_transaction(
        self,
        txn_dict: Dict[str, Any],
        raw_xgb_prob: float,
        raw_gnn_prob: float,
        top_k_features: int = 5
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive explainability report for a single transaction:
        1. Calibrated hybrid risk score and risk tier.
        2. Recommended operational action.
        3. Native TreeSHAP feature attributions for tabular risk.
        4. Relational graph topological drivers.
        5. Prominent limitation disclosures and caveats.
        """
        cal_xgb = float(self.calibrate_xgb(np.array([raw_xgb_prob]))[0])
        cal_gnn = float(self.calibrate_gnn(np.array([raw_gnn_prob]))[0])
        hybrid_score = float(self.weight_xgb * cal_xgb + self.weight_gnn * cal_gnn)
        tier = self.classify_tier(hybrid_score)
        action = self.ACTION_MAP[tier]

        # 1. Tabular SHAP explanations via native XGBoost Booster
        shap_explanations = []
        if self.feature_pipeline is not None and self.xgb_model is not None:
            df_single = pd.DataFrame([txn_dict])
            feat_df = self.feature_pipeline.transform(df_single)
            booster = self.xgb_model.get_booster()
            dmat = xgb.DMatrix(feat_df, feature_names=self.feature_pipeline.feature_names_)
            contribs = booster.predict(dmat, pred_contribs=True)[0]
            feature_names = self.feature_pipeline.feature_names_

            # Last value in contribs is expected value / bias
            feat_contribs = contribs[:-1]
            bias = float(contribs[-1])

            sorted_indices = np.argsort(np.abs(feat_contribs))[::-1]
            for idx in sorted_indices[:top_k_features]:
                fname = feature_names[idx]
                fval = float(feat_df.iloc[0, idx])
                val_contrib = float(feat_contribs[idx])
                shap_explanations.append({
                    "feature": fname,
                    "value": round(fval, 4),
                    "shap_contribution": round(val_contrib, 4),
                    "impact": "INCREASES_FRAUD_RISK" if val_contrib > 0 else "REDUCES_FRAUD_RISK"
                })

        # 2. Graph Topological Drivers
        graph_drivers = {
            "account_orig": str(txn_dict.get("nameOrig", "UNKNOWN")),
            "account_dest": str(txn_dict.get("nameDest", "UNKNOWN")),
            "is_destination_merchant": str(txn_dict.get("nameDest", "")).startswith("M"),
            "raw_gnn_anomaly_score": round(raw_gnn_prob, 4),
            "calibrated_gnn_risk": round(cal_gnn, 4),
            "relational_weight_in_ensemble": round(self.weight_gnn, 2)
        }

        if self.graph_builder is not None:
            u_id = self.graph_builder.node_to_idx.get(str(txn_dict.get("nameOrig")), None)
            v_id = self.graph_builder.node_to_idx.get(str(txn_dict.get("nameDest")), None)
            graph_drivers["orig_historical_out_degree"] = self.graph_builder.out_degrees.get(u_id, 0.0) if u_id is not None else 0.0
            graph_drivers["orig_historical_out_volume"] = round(self.graph_builder.out_volume.get(u_id, 0.0), 2) if u_id is not None else 0.0
            graph_drivers["dest_historical_in_degree"] = self.graph_builder.in_degrees.get(v_id, 0.0) if v_id is not None else 0.0
            graph_drivers["dest_historical_in_volume"] = round(self.graph_builder.in_volume.get(v_id, 0.0), 2) if v_id is not None else 0.0

        return {
            "transaction_summary": {
                "step": int(txn_dict.get("step", 0)),
                "type": str(txn_dict.get("type", "")),
                "amount": float(txn_dict.get("amount", 0.0)),
                "oldbalanceOrg": float(txn_dict.get("oldbalanceOrg", 0.0)),
                "newbalanceOrig": float(txn_dict.get("newbalanceOrig", 0.0))
            },
            "risk_assessment": {
                "hybrid_risk_score": round(hybrid_score, 4),
                "assigned_risk_tier": tier,
                "recommended_action": action,
                "score_components": {
                    "tabular_xgb": {
                        "raw_score": round(raw_xgb_prob, 4),
                        "calibrated_probability": round(cal_xgb, 4),
                        "ensemble_weight": round(self.weight_xgb, 2),
                        "weighted_contribution": round(self.weight_xgb * cal_xgb, 4)
                    },
                    "relational_gnn": {
                        "raw_score": round(raw_gnn_prob, 4),
                        "calibrated_probability": round(cal_gnn, 4),
                        "ensemble_weight": round(self.weight_gnn, 2),
                        "weighted_contribution": round(self.weight_gnn * cal_gnn, 4)
                    }
                }
            },
            "tabular_shap_attributions": shap_explanations,
            "graph_topological_context": graph_drivers,
            "regulatory_disclaimers": self.DISCLAIMERS
        }

    def save_configuration(self, config_path: Path):
        """Saves ensemble weights, thresholds, and tier metadata to JSON."""
        config_data = {
            "ensemble_architecture": "Sentinel Calibrated Hybrid Ensemble",
            "weights": {
                "tabular_xgboost": self.weight_xgb,
                "relational_graphsage": self.weight_gnn
            },
            "risk_tier_thresholds": {
                "critical": self.threshold_critical,
                "high": self.threshold_high,
                "medium": self.threshold_medium,
                "low": self.threshold_low
            },
            "tier_actions": self.ACTION_MAP,
            "disclaimers": self.DISCLAIMERS
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def load(
        cls,
        config_path: Path,
        calibrators_path: Path,
        pipeline_path: Optional[Path] = None,
        xgb_path: Optional[Path] = None,
        gnn_weights_path: Optional[Path] = None
    ) -> "SentinelHybridRiskModel":
        """Loads a configured SentinelHybridRiskModel from disk."""
        with open(config_path, "r") as f:
            cfg = json.load(f)

        calibrators = joblib.load(calibrators_path)
        cal_xgb = calibrators["calibrator_xgb"]
        cal_gnn = calibrators["calibrator_gnn"]

        feature_pipeline = None
        if pipeline_path and pipeline_path.exists():
            feature_pipeline = joblib.load(pipeline_path)

        xgb_model = None
        if xgb_path and xgb_path.exists():
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(xgb_path)

        weights_dict = cfg.get("ensemble_weights", cfg.get("weights", {}))
        return cls(
            weight_xgb=weights_dict.get("tabular_xgboost", 0.80),
            weight_gnn=weights_dict.get("relational_graphsage", 0.20),
            threshold_critical=cfg["risk_tier_thresholds"]["critical"],
            threshold_high=cfg["risk_tier_thresholds"]["high"],
            threshold_medium=cfg["risk_tier_thresholds"]["medium"],
            calibrator_xgb=cal_xgb,
            calibrator_gnn=cal_gnn,
            feature_pipeline=feature_pipeline,
            xgb_model=xgb_model
        )
