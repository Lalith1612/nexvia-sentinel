"""
Sentinel Graph Builder Module (Strictly Temporal & Leakage-Free)
Constructs directed temporal account graphs from PaySim transactions for PyTorch Geometric.
Guarantees:
1. Strict 1-to-1 row alignment with input DataFrames (zero index shuffling).
2. Zero temporal leakage: running degrees and volumes are computed strictly prior to transaction execution.
3. Continuous historical streaming state across Train -> Val -> Test partitions.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

NODE_FEATURE_NAMES = [
    "is_merchant",
    "log_prior_in_degree",
    "log_prior_out_degree",
    "log_prior_in_volume",
    "log_prior_out_volume"
]

EDGE_FEATURE_NAMES = [
    "log_amount",
    "log_oldbalance_orig",
    "log_oldbalance_dest",
    "amount_to_oldbalance_ratio",
    "hour_of_day",
    "day_of_week",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "src_prior_out_degree",
    "src_prior_out_volume",
    "dst_prior_in_degree",
    "dst_prior_in_volume"
]

class SentinelGraphBuilder:
    """
    Constructs and indexes temporal directed account graphs from raw transaction records.
    Maintains running account ledger state across sequential partitions without future leakage.
    """

    def __init__(self):
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: List[str] = []
        # Running temporal state: account -> historical counters strictly before current transaction
        self.in_degrees: Dict[int, float] = {}
        self.out_degrees: Dict[int, float] = {}
        self.in_volume: Dict[int, float] = {}
        self.out_volume: Dict[int, float] = {}
        self.is_fitted: bool = False

    def get_or_create_node_id(self, account_str: str) -> int:
        if account_str not in self.node_to_idx:
            idx = len(self.idx_to_node)
            self.node_to_idx[account_str] = idx
            self.idx_to_node.append(account_str)
            return idx
        return self.node_to_idx[account_str]

    def build_graph_data(
        self,
        df: pd.DataFrame,
        update_state: bool = True
    ) -> Tuple[Data, pd.DataFrame, Dict]:
        """
        Builds a PyTorch Geometric Data object from transactions preserving exact row order.
        Returns: (Data, df_aligned, metadata) where edge i matches df_aligned.iloc[i] exactly.
        """
        # Ensure contiguous 0..N-1 index without re-ordering rows
        df_aligned = df.reset_index(drop=True)
        n_txns = len(df_aligned)
        print(f"[GRAPH_BUILDER] Building graph from {n_txns:,} transactions (preserving 100% row order)...")

        src_nodes = df_aligned["nameOrig"].astype(str).values
        dst_nodes = df_aligned["nameDest"].astype(str).values
        amounts = df_aligned["amount"].values.astype(np.float32)

        src_indices = np.zeros(n_txns, dtype=np.int64)
        dst_indices = np.zeros(n_txns, dtype=np.int64)

        # Edge prior state arrays (strictly before transaction i executes)
        src_prior_out_deg = np.zeros(n_txns, dtype=np.float32)
        src_prior_out_vol = np.zeros(n_txns, dtype=np.float32)
        dst_prior_in_deg = np.zeros(n_txns, dtype=np.float32)
        dst_prior_in_vol = np.zeros(n_txns, dtype=np.float32)

        for i in range(n_txns):
            u = self.get_or_create_node_id(src_nodes[i])
            v = self.get_or_create_node_id(dst_nodes[i])
            src_indices[i] = u
            dst_indices[i] = v

            # Record prior state before this transaction
            src_prior_out_deg[i] = self.out_degrees.get(u, 0.0)
            src_prior_out_vol[i] = self.out_volume.get(u, 0.0)
            dst_prior_in_deg[i] = self.in_degrees.get(v, 0.0)
            dst_prior_in_vol[i] = self.in_volume.get(v, 0.0)

            # Update running state if requested
            if update_state:
                amt_i = float(amounts[i])
                self.out_degrees[u] = self.out_degrees.get(u, 0.0) + 1.0
                self.out_volume[u] = self.out_volume.get(u, 0.0) + amt_i
                self.in_degrees[v] = self.in_degrees.get(v, 0.0) + 1.0
                self.in_volume[v] = self.in_volume.get(v, 0.0) + amt_i

        n_unique_nodes = len(self.idx_to_node)
        print(f"  Indexed {n_unique_nodes:,} total unique account nodes.")

        # 2. Node Features
        is_merchant = np.array([1.0 if n.startswith("M") else 0.0 for n in self.idx_to_node], dtype=np.float32)
        node_in_deg = np.array([self.in_degrees.get(idx, 0.0) for idx in range(n_unique_nodes)], dtype=np.float32)
        node_out_deg = np.array([self.out_degrees.get(idx, 0.0) for idx in range(n_unique_nodes)], dtype=np.float32)
        node_in_vol = np.array([self.in_volume.get(idx, 0.0) for idx in range(n_unique_nodes)], dtype=np.float32)
        node_out_vol = np.array([self.out_volume.get(idx, 0.0) for idx in range(n_unique_nodes)], dtype=np.float32)

        node_features = np.column_stack([
            is_merchant,
            np.log1p(node_in_deg),
            np.log1p(node_out_deg),
            np.log1p(node_in_vol),
            np.log1p(node_out_vol)
        ]).astype(np.float32)

        # 3. Edge Attributes (15 features, including prior running state)
        amt_log = np.log1p(np.maximum(0.0, amounts))
        orig_bal_log = np.log1p(np.maximum(0.0, df_aligned["oldbalanceOrg"].values.astype(np.float32)))
        dest_bal_log = np.log1p(np.maximum(0.0, df_aligned["oldbalanceDest"].values.astype(np.float32)))
        amt_ratio = amounts / (df_aligned["oldbalanceOrg"].values.astype(np.float32) + 1.0)

        steps = df_aligned["step"].values.astype(np.int64)
        hour = (steps % 24).astype(np.float32)
        day = ((steps // 24) % 7).astype(np.float32)

        tx_types = df_aligned["type"].values.astype(str)
        type_encs = [(tx_types == t).astype(np.float32) for t in TRANSACTION_TYPES]

        edge_features = np.column_stack([
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
        ]).astype(np.float32)

        labels = df_aligned["isFraud"].values.astype(np.float32)

        # PyG Data Object
        edge_index = torch.tensor(np.vstack([src_indices, dst_indices]), dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.float)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.float)
        edge_steps = torch.tensor(steps, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            edge_step=edge_steps,
            num_nodes=n_unique_nodes
        )

        metadata = {
            "num_nodes": n_unique_nodes,
            "num_edges": n_txns,
            "num_node_features": len(NODE_FEATURE_NAMES),
            "num_edge_features": len(EDGE_FEATURE_NAMES),
            "fraud_edges": int(np.sum(labels == 1.0)),
            "legit_edges": int(np.sum(labels == 0.0)),
            "fraud_prevalence_pct": round(float(np.mean(labels)) * 100, 4)
        }

        self.is_fitted = True
        return data, df_aligned, metadata

    def get_current_node_features(self) -> torch.Tensor:
        """
        Builds the node feature tensor strictly from currently settled historical counters.
        Guarantees that no future transactions leak into node features.
        """
        n_unique_nodes = len(self.idx_to_node)
        is_merchant = np.array([1.0 if n.startswith("M") else 0.0 for n in self.idx_to_node], dtype=np.float32)
        node_in_deg = np.array([self.in_degrees.get(idx, 0.0) for idx in range(n_unique_nodes)], dtype=np.float32)
        node_out_deg = np.array([self.out_degrees.get(idx, 0.0) for idx in range(n_unique_nodes)], dtype=np.float32)
        node_in_vol = np.array([self.in_volume.get(idx, 0.0) for idx in range(n_unique_nodes)], dtype=np.float32)
        node_out_vol = np.array([self.out_volume.get(idx, 0.0) for idx in range(n_unique_nodes)], dtype=np.float32)

        node_features = np.column_stack([
            is_merchant,
            np.log1p(node_in_deg),
            np.log1p(node_out_deg),
            np.log1p(node_in_vol),
            np.log1p(node_out_vol)
        ]).astype(np.float32)
        return torch.tensor(node_features, dtype=torch.float)

    def build_step_candidate_edges(self, df_step: pd.DataFrame, update_state: bool = False):
        """
        Processes candidate transactions for a specific time step or window.
        1. Maps accounts to stable contiguous indices (creating new ones if unseen).
        2. Queries pre-authorization prior degree and volume strictly BEFORE these transactions settle.
        3. Constructs the 15 edge attributes.
        4. If update_state is True, updates running counters AFTER recording all prior attributes.
        Returns: (q_edge, q_attr, labels, df_aligned)
        """
        df_aligned = df_step.reset_index(drop=True)
        n_txns = len(df_aligned)
        src_nodes = df_aligned["nameOrig"].astype(str).values
        dst_nodes = df_aligned["nameDest"].astype(str).values
        amounts = df_aligned["amount"].values.astype(np.float32)

        src_indices = np.zeros(n_txns, dtype=np.int64)
        dst_indices = np.zeros(n_txns, dtype=np.int64)

        src_prior_out_deg = np.zeros(n_txns, dtype=np.float32)
        src_prior_out_vol = np.zeros(n_txns, dtype=np.float32)
        dst_prior_in_deg = np.zeros(n_txns, dtype=np.float32)
        dst_prior_in_vol = np.zeros(n_txns, dtype=np.float32)

        for i in range(n_txns):
            u = self.get_or_create_node_id(src_nodes[i])
            v = self.get_or_create_node_id(dst_nodes[i])
            src_indices[i] = u
            dst_indices[i] = v

            src_prior_out_deg[i] = self.out_degrees.get(u, 0.0)
            src_prior_out_vol[i] = self.out_volume.get(u, 0.0)
            dst_prior_in_deg[i] = self.in_degrees.get(v, 0.0)
            dst_prior_in_vol[i] = self.in_volume.get(v, 0.0)

        if update_state:
            for i in range(n_txns):
                u = src_indices[i]
                v = dst_indices[i]
                amt_i = float(amounts[i])
                self.out_degrees[u] = self.out_degrees.get(u, 0.0) + 1.0
                self.out_volume[u] = self.out_volume.get(u, 0.0) + amt_i
                self.in_degrees[v] = self.in_degrees.get(v, 0.0) + 1.0
                self.in_volume[v] = self.in_volume.get(v, 0.0) + amt_i

        amt_log = np.log1p(np.maximum(0.0, amounts))
        orig_bal_log = np.log1p(np.maximum(0.0, df_aligned["oldbalanceOrg"].values.astype(np.float32)))
        dest_bal_log = np.log1p(np.maximum(0.0, df_aligned["oldbalanceDest"].values.astype(np.float32)))
        amt_ratio = amounts / (df_aligned["oldbalanceOrg"].values.astype(np.float32) + 1.0)

        steps = df_aligned["step"].values.astype(np.int64)
        hour = (steps % 24).astype(np.float32)
        day = ((steps // 24) % 7).astype(np.float32)

        tx_types = df_aligned["type"].values.astype(str)
        type_encs = [(tx_types == t).astype(np.float32) for t in TRANSACTION_TYPES]

        edge_features = np.column_stack([
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
        ]).astype(np.float32)

        labels = df_aligned["isFraud"].values.astype(np.float32) if "isFraud" in df_aligned.columns else np.zeros(n_txns, dtype=np.float32)

        q_edge = torch.tensor(np.vstack([src_indices, dst_indices]), dtype=torch.long)
        q_attr = torch.tensor(edge_features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.float)

        return q_edge, q_attr, y, df_aligned

    def settle_step_transactions(self, q_edge: torch.Tensor, amounts: np.ndarray):
        """
        Settles candidate transactions into the historical graph state.
        Must be called strictly AFTER predictions have been generated for step s.
        """
        n_txns = q_edge.shape[1]
        src_indices = q_edge[0].numpy()
        dst_indices = q_edge[1].numpy()
        for i in range(n_txns):
            u = int(src_indices[i])
            v = int(dst_indices[i])
            amt_i = float(amounts[i])
            self.out_degrees[u] = self.out_degrees.get(u, 0.0) + 1.0
            self.out_volume[u] = self.out_volume.get(u, 0.0) + amt_i
            self.in_degrees[v] = self.in_degrees.get(v, 0.0) + 1.0
            self.in_volume[v] = self.in_volume.get(v, 0.0) + amt_i

    def save_state(self, filepath: Path):
        with open(filepath, "w") as f:
            json.dump({
                "node_to_idx": self.node_to_idx,
                "in_degrees": {str(int(k)): float(v) for k, v in self.in_degrees.items()},
                "out_degrees": {str(int(k)): float(v) for k, v in self.out_degrees.items()},
                "in_volume": {str(int(k)): float(v) for k, v in self.in_volume.items()},
                "out_volume": {str(int(k)): float(v) for k, v in self.out_volume.items()}
            }, f)

    @classmethod
    def load_state(cls, filepath: Path) -> "SentinelGraphBuilder":
        builder = cls()
        with open(filepath, "r") as f:
            data = json.load(f)
            builder.node_to_idx = data["node_to_idx"]
            builder.idx_to_node = [None] * len(builder.node_to_idx)
            for k, v in builder.node_to_idx.items():
                builder.idx_to_node[v] = k
            builder.in_degrees = {int(k): float(v) for k, v in data.get("in_degrees", {}).items()}
            builder.out_degrees = {int(k): float(v) for k, v in data.get("out_degrees", {}).items()}
            builder.in_volume = {int(k): float(v) for k, v in data.get("in_volume", {}).items()}
            builder.out_volume = {int(k): float(v) for k, v in data.get("out_volume", {}).items()}
            builder.is_fitted = True
        return builder
