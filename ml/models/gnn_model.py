"""
Sentinel GraphSAGE Neural Network Architecture
Edge-conditioned inductive Graph Neural Network for fraud detection and syndicate identification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class SentinelGraphSAGE(nn.Module):
    """
    Inductive 2-layer GraphSAGE architecture with edge-conditioned fraud classifier head.
    Node Encoder: SAGEConv(5 -> 64) -> LayerNorm -> ReLU -> Dropout -> SAGEConv(64 -> 32)
    Edge MLP: [h_src (32) || h_dst (32) || e_attr (11)] -> Linear(75 -> 64) -> ReLU -> Linear(64 -> 1)
    """

    def __init__(
        self,
        node_in_channels: int = 5,
        edge_in_channels: int = 11,
        hidden_channels: int = 64,
        out_channels: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_p = dropout

        # Node representation layers
        self.conv1 = SAGEConv(node_in_channels, hidden_channels, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean")
        self.norm2 = nn.LayerNorm(out_channels)

        # Edge classification MLP
        edge_mlp_in = (out_channels * 2) + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Computes structural node embeddings via neighborhood aggregation."""
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_p, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.relu(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        query_edge_index: torch.Tensor = None,
        query_edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass producing raw logits for edges.
        If query_edge_index is None, evaluates on edge_index.
        """
        # 1. Compute node embeddings
        node_emb = self.encode_nodes(x, edge_index)

        # 2. Select query edges
        if query_edge_index is None:
            q_src = edge_index[0]
            q_dst = edge_index[1]
            q_attr = edge_attr
        else:
            q_src = query_edge_index[0]
            q_dst = query_edge_index[1]
            q_attr = query_edge_attr

        # 3. Concatenate [h_u || h_v || e_uv]
        h_src = node_emb[q_src]
        h_dst = node_emb[q_dst]
        edge_repr = torch.cat([h_src, h_dst, q_attr], dim=-1)

        # 4. Predict edge logits
        logits = self.edge_mlp(edge_repr).squeeze(-1)
        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        query_edge_index: torch.Tensor = None,
        query_edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        """Returns calibrated fraud probabilities in [0.0, 1.0]."""
        with torch.no_grad():
            logits = self.forward(x, edge_index, edge_attr, query_edge_index, query_edge_attr)
            return torch.sigmoid(logits)

    def get_account_risk_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Computes account-level maximum fraud risk across connected transaction edges."""
        with torch.no_grad():
            edge_probs = self.predict_proba(x, edge_index, edge_attr)
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            num_nodes = x.size(0)

            node_max_risk = torch.zeros(num_nodes, dtype=torch.float)
            # Scatter max
            for i in range(edge_probs.size(0)):
                p = edge_probs[i].item()
                u = src_nodes[i].item()
                v = dst_nodes[i].item()
                if p > node_max_risk[u]:
                    node_max_risk[u] = p
                if p > node_max_risk[v]:
                    node_max_risk[v] = p

            return node_max_risk
