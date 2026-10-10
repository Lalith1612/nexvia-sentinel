# Sentinel Architecture & Engineering Specification

## 1. Overview
Sentinel is an end-to-end, real-time financial fraud detection engine designed for modern fintech infrastructure. It fuses tabular classification (XGBoost) with relational graph neural network analysis (PyTorch Geometric) over a streaming event architecture (Apache Kafka).

## 2. Core Subsystems

### 2.1 Tabular Anomaly Engine (XGBoost)
- **Role:** Evaluates individual transaction-level risk indicators in < 5ms.
- **Key Indicators:** Balance error deltas, transaction-to-balance ratios, zero-balance liquidation anomalies, temporal transaction velocity.
- **Objective Function:** Binary logistic with class-weight calibration (`scale_pos_weight`) to optimize PR-AUC under severe fraud class rarity (~0.13%).

### 2.2 Relational Graph Engine (PyTorch Geometric)
- **Role:** Detects syndicates, mule networks, cyclic laundering, and high-degree hubs.
- **Graph Topology:** Accounts represented as graph nodes, transactions as directed edges with timestamp and amount attributes.
- **Model Architecture:** Multi-layer GraphSAGE / Relational GCN learning inductive node embeddings to score edge and neighborhood fraud probability.

### 2.3 Streaming & Feature Serving
- **Message Broker:** Apache Kafka topic `sentinel.transactions`.
- **Feature Store:** Feast online feature store maintaining sub-millisecond account aggregate counters (e.g. rolling 1-hour transaction volume, average amount, recipient diversity).

### 2.4 Hybrid Risk Calibration
$$Risk_{composite} = \alpha \cdot P_{XGB}(Fraud) + (1 - \alpha) \cdot P_{GNN}(Fraud)$$
- Configurable dynamic weights $\alpha \in [0, 1]$.
- Decision thresholds:
  - `CRITICAL`: $\ge 0.85$ (Automated freeze recommendation)
  - `HIGH`: $0.65 - 0.85$ (Priority analyst triage)
  - `MEDIUM`: $0.40 - 0.65$ (Secondary authentication required)
  - `LOW`: $< 0.40$ (Approved transaction)

### 2.5 Observability & Operations
- **Experiment Tracking:** MLflow tracking parameters, PR curves, and versioned artifacts.
- **Drift Monitoring:** Evidently AI monitoring distribution drift across key features (`amount`, balance error) and alert rates over time.
- **Analyst Feedback Loop:** Triage decisions (CONFIRMED_FRAUD, FALSE_POSITIVE) recorded in PostgreSQL with audit history for continuous retraining.
