# Sentinel — Account Graph Construction & GNN Fraud Detection Specification

## 1. Graph Topology & Entity Definitions

To detect syndicated financial crime, Sentinel models the financial transaction universe as a directed temporal multigraph $G = (V, E, \mathcal{T})$:

### 1.1 Nodes ($V$ — Accounts)
- **Node Identifier**: Account ID (`nameOrig`, `nameDest`).
- **Entity Classes**:
  - `Customer` (prefix `C*`): Personal mobile-money or retail banking customer accounts.
  - `Merchant / Agent` (prefix `M*`): Business accounts and cash-in/cash-out retail kiosks.
- **Node Features ($\mathbf{x}_u \in \mathbb{R}^{d_v}$)**:
  - `is_merchant`: Binary indicator (1 for `M*`, 0 for `C*`).
  - `in_degree_hist`: Historical count of inward transactions.
  - `out_degree_hist`: Historical count of outward transactions.
  - `in_volume_hist`: Cumulative currency received prior to current timestamp.
  - `out_volume_hist`: Cumulative currency dispatched prior to current timestamp.

### 1.2 Edges ($E$ — Transactions)
- **Directed Edge**: Directed link $e = (u, v)$ from sender account $u$ (`nameOrig`) to beneficiary account $v$ (`nameDest`).
- **Edge Attributes ($\mathbf{e}_{uv} \in \mathbb{R}^{d_e}$)**:
  - `log_amount`: Log-scaled transaction value $\log(1 + \text{amount})$.
  - `log_oldbalance_orig`: Sender balance prior to transaction.
  - `log_oldbalance_dest`: Beneficiary balance prior to transaction.
  - `amount_to_balance_ratio`: Ratio $\frac{\text{amount}}{\text{oldbalanceOrg} + 1.0}$.
  - `type_one_hot`: 5-dimensional binary vector (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT`).
  - `hour_of_day`: Diurnal hour ($step \pmod{24}$).
  - `day_of_week`: Weekly day ($\lfloor step / 24\rfloor \pmod 7$).
  - `timestamp_step`: Simulation step hour.

---

## 2. Formulation of the Prediction Task

### 2.1 Transaction-Level vs Account-Level Detection
- **Transaction-Level Classification (Primary Task)**:
  Evaluates whether an individual transaction edge $e_{uv}$ occurring at authorization time is fraudulent ($y_e \in \{0, 1\}$).
- **Account-Level Risk Scoring (Derived Metric)**:
  Measures the aggregate probability that an account node $u$ acts as a money mule, stolen account, or ring hub:
  $$\text{AccountRisk}(u) = \max_{v \in \mathcal{N}(u)} P(\text{Fraud}(e_{uv}))$$

### 2.2 Temporal Leakage Prevention Protocol
1. **Zero Future Edges**: An edge at simulation step $t$ only aggregates messages from prior edges $e' \in E$ where $\text{step}(e') \le t$.
2. **Zero Label Contamination**: Ground truth labels ($isFraud$) are **never** passed as node features or edge attributes during message passing.
3. **Partition Isolation**: The graph is constructed sequentially. Training evaluates on edges with $step \le 500$, validation on $501 \le step \le 620$, and test on $621 \le step \le 743$.

---

## 3. PyTorch Geometric Architecture: Sentinel GraphSAGE

Sentinel implements an inductive **Edge-Conditioned GraphSAGE** architecture:

```text
[Sender Account u]  ──> SAGEConv Layer 1 (Mean Aggr) ──> LayerNorm + ReLU ──> SAGEConv Layer 2 ──> h_u (32-dim)
                                                                                                        │
[Receiver Account v]──> SAGEConv Layer 1 (Mean Aggr) ──> LayerNorm + ReLU ──> SAGEConv Layer 2 ──> h_v (32-dim)
                                                                                                        │
[Edge Features e_uv]──────────────────────────────────────────────────────────────────────────────> e_uv (12-dim)
                                                                                                        │
                                                                                                        ▼
                                                                                   Concat: [h_u || h_v || e_uv] (76-dim)
                                                                                                        │
                                                                                                        ▼
                                                                                   MLP Classifier: Linear(76->64) -> ReLU
                                                                                                   Linear(64->1)  -> Sigmoid
                                                                                                        │
                                                                                                        ▼
                                                                                         P(Fraud | Graph Neighborhood)
```

### 3.1 Loss Function & Class Imbalance
Trained with `BCEWithLogitsLoss` incorporating positive class weighting:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ w_{\text{pos}} \cdot y_i \log(\sigma(\hat{y}_i)) + (1 - y_i) \log(1 - \sigma(\hat{y}_i)) \right]$$
where $w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$.

---

## 4. Memory & Resource Budget (8 GB RAM Host)
- Graph node mapping assigns contiguous 32-bit integers to unique accounts.
- Peak tensor memory is budgeted to **< 300 MB**, well within the 8 GB machine capacity.
- Execution leverages PyTorch CPU multi-threading, completing full training in under 2 minutes.
