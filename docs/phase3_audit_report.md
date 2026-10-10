# Sentinel — PyTorch Geometric GraphSAGE Evaluation & Audit Report

## 1. Ground Truth Reconciliation & Discrepancy Resolution
In this audit, all ground truth counts were verified directly against untouched `test.parquet`:

| Transaction Type | Total Transactions | Verified Fraud Cases | Fraud Rate | Initial Reported Frauds | Audit Reconciliation Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TRANSFER** | 8,974 | **683** | 7.611% | 499 | **RECONCILED (Discrepancy Resolved)** |
| **CASH_OUT** | 29,501 | **683** | 2.315% | 586 | **RECONCILED (Discrepancy Resolved)** |
| **PAYMENT** | 30,646 | **0** | 0.000% | 152 | **RECONCILED (Confirmed 0 Frauds)** |
| **CASH_IN** | 20,993 | **0** | 0.000% | 121 | **RECONCILED (Confirmed 0 Frauds)** |
| **DEBIT** | 715 | **0** | 0.000% | 8 | **RECONCILED (Confirmed 0 Frauds)** |
| **TOTAL** | **90,829** | **1,366** | **1.5039%** | 1,366 | **100% RECONCILED (683 + 683 = 1,366)** |

### Explanation of Discrepancy:
In the initial script, an internal call to `sort_values` altered the row order within step hours. Slicing masks generated from the un-sorted DataFrame misaligned with the model's output array. The pipeline now enforces **strict row-order preservation**, verified with an automated assertion.

---

## 2. Reconciled GNN Performance Metrics on Holdout Test Set

| Metric | GraphSAGE GNN (Phase 3 Causal) | Tabular XGBoost (Phase 2) | Majority Baseline |
| :--- | :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | **0.6366** | 0.9993 | 0.0150 |
| **ROC-AUC** | **0.9669** | 0.9994 | 0.5000 |
| **Operational Threshold** | **0.9** | 0.620 | 0.500 |
| **Precision** | **80.05%** | 99.93% | 0.00% |
| **Recall (Detection Rate)** | **44.36%** | 99.93% | 0.00% |
| **F1-Score** | **0.5709** | 0.9993 | 0.0000 |

### Confusion Matrix (Threshold = 0.9):
$$\begin{bmatrix} \text{TN} = 89,312 & \text{FP} = 151 \\ \text{FN} = 760 & \text{TP} = 606 \end{bmatrix}$$

![PR Curve](figures/11_gnn_precision_recall_curve.png)
![Confusion Matrix](figures/12_gnn_confusion_matrix.png)

---

## 3. Reconciled Performance Slices by Transaction Type

| Type | Total Transactions | Actual Frauds | PR-AUC | Precision | Recall | FPR | TP | FP | FN | TN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRANSFER** | 8,974 | **683** | 0.9304 | 94.32% | 63.25% | 0.3136% | 432 | 26 | 251 | 8,265 |
| **CASH_OUT** | 29,501 | **683** | 0.3923 | 58.19% | 25.48% | 0.4338% | 174 | 125 | 509 | 28,693 |
| **PAYMENT** | 30,646 | **0** | N/A (0 ground truth frauds) | N/A (0 predicted positives) | N/A (0 ground truth frauds) | 0.0000% | 0 | 0 | 0 | 30,646 |
| **CASH_IN** | 20,993 | **0** | N/A (0 ground truth frauds) | N/A (0 predicted positives) | N/A (0 ground truth frauds) | 0.0000% | 0 | 0 | 0 | 20,993 |
| **DEBIT** | 715 | **0** | N/A (0 ground truth frauds) | N/A (0 predicted positives) | N/A (0 ground truth frauds) | 0.0000% | 0 | 0 | 0 | 715 |

---

## 4. Fair Comparable Latency Benchmarks (Cached vs Uncached Profiles)

All benchmarks measured on Windows 11 CPU with identical warmups:

| Evaluation Scope | Pipeline Stage | XGBoost Tabular (Complete End-to-End) | GraphSAGE GNN (Partial-Path Cached Scoring) | GraphSAGE GNN (Full-Graph CPU Inference) | Latency Measurement Boundaries & Architectural Scope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Single Txn (Batch = 1)** | Feature Extraction / Real Edge Attributes | 17.381 ms | 0.07 ms | 0.131 ms | Real 15-attribute candidate feature construction and node ID lookup. |
| | Embedding Retrieval / Message Passing | N/A | 0.015 ms | 607.645 ms | Cached: in-memory snapshot retrieval.<br>Uncached: synchronous full-graph message passing on CPU across active graph. |
| | Scoring Head (Edge MLP / Trees) | **1.106 ms** | **0.169 ms** | **0.347 ms** | Tree inference vs. PyTorch Edge MLP forward pass. |
| | **Measured Total** | **18.487 ms** | **0.253 ms** | **608.123 ms** | **Scope Finding:** Cached GNN measures **partial-path online scoring** given precomputed embeddings. Uncached full-graph CPU GNN exceeds strict SLAs (<50 ms). (Dynamic neighborhood sampling is not implemented). |
| **Batch Inference (1,000 txns)** | Inference Throughput | 7.12 ms | 2.29 ms | N/A | Vectorized batch pass across candidate edges. |
| | **Per-Transaction Latency** | **0.0167 ms/txn** | **0.0023 ms/txn** | N/A | High throughput under batch vectorization. |

![Model Comparison](figures/13_xgboost_vs_gnn_comparison.png)

---

## 5. Temporal Protocol Disclosure: Causal Training Windows vs Streaming Evaluation

| Protocol Attribute | Causal Training Protocol | Streaming Test Evaluation Protocol | Architectural Rationale & Distinctions |
| :--- | :--- | :--- | :--- |
| **Temporal Granularity** | 20 windows of 25 hours each (steps 1–500) | 123 hourly streaming steps (steps 621–743) | Training batches over 25 hours for GPU/CPU memory stability and gradient optimization; evaluation simulates true hourly streaming arrivals. |
| **Message Passing Edges** | Strictly settled edges prior to window ($t < w_{start}$) | Strictly settled edges prior to step ($t < s$) | Both enforce $t_{pred} > t_{graph}$ strictly. |
| **Node Feature State** | Historical degree/volume counters prior to window | Historical degree/volume counters prior to step | Guaranteed zero future-transaction leakage into node features. |
| **Historical Graph Seeding** | Starts empty at step 1; accumulates across windows | Seeded with settled history through step 620 | Test evaluation evaluates on realistic, settled enterprise topology. |

> **Critical Note:** The 25-hour causal training windows and hourly streaming evaluation are **distinct protocols** optimized for their respective objectives (gradient descent memory management vs streaming inference). They do not share identical time granularities, but both strictly maintain the causal invariant.

---

## 6. Topology Audit & Conceptual Diagram Clarification
- Discovered 2-hop mule accounts in test partition: **0**.
- **Figure 14 is a Conceptual Domain Architecture Archetype**, illustrating how relational fraud flows occur in physical mobile-money networks. In PaySim synthetic data, transactions are generated as disjoint pairs, meaning accounts do not participate in multi-hop chains within the simulation. Ring detection is not claimed on PaySim.

![Conceptual Architecture](figures/14_sample_fraud_ring_subgraph.png)
