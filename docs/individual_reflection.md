# Nexvia Sentinel — Individual Engineering Reflection (AIML-01)

**Author:** Lalith / Nexvia AI/ML Internship
**Project:** Nexvia Sentinel — Real-Time Financial Fraud Detection Platform
**Repository:** `Nexvia/AIML-01`

---

## 1. Introduction & Context

During the Nexvia AIML-01 project, my objective was to design, implement, and deploy an enterprise-grade financial fraud detection engine for digital payments.

Rather than treating fraud detection as an isolated tabular classification exercise, I designed Nexvia Sentinel as a production-oriented platform fusing high-speed tabular gradient boosting (**XGBoost**), inductive relational graph neural networks (**PyTorch Geometric GraphSAGE**), automated drift monitoring (**Evidently AI**), experiment tracking (**MLflow**), feature store sandboxing (**Feast**), and cloud deployment on Render and Vercel.

This reflection outlines the technical decisions, engineering trade-offs, methodological realizations, and architectural lessons encountered throughout the project.

---

## 2. Technical Architecture & Hybrid Modeling Decisions

### Why a Dual-Engine Hybrid Architecture?
In modern payment processing, fraud manifests in two distinct patterns:
1. **Isolated Transaction Anomalies:** Sudden liquidation of account balances, abnormal transfer velocities, or unusual transaction amounts relative to historical balances.
2. **Coordinated Syndicated Fraud Rings:** Money-mule rings, cyclical laundering, and rapid multi-hop disbursement where individual transactions are deliberately kept small to evade static amount-based rules.

A purely tabular model (like XGBoost) evaluates each transaction in isolation. It lacks structural memory of account relationships and cannot identify whether an account is acting as an intermediary node in a laundering chain. Conversely, a pure Graph Neural Network (GNN) excels at neighborhood aggregation but has higher computational complexity and can be less sensitive to fine-grained numerical arithmetic deltas.

To solve this, I designed a hybrid ensemble:
$$Risk_{composite} = \alpha \cdot P_{XGB}(Fraud) + (1 - \alpha) \cdot P_{GNN}(Fraud)$$
Using Platt scaling calibration and validation tuning, I settled on $\alpha = 0.78$ for XGBoost and $0.22$ for GraphSAGE. This allows tabular gradient boosting to handle primary transaction-level indicators within 12ms, while GraphSAGE injects multi-hop structural context.

---

## 3. Tackling Severe Class Imbalance & Metric Selection

With only 8,213 frauds out of 6.36 million PaySim transactions (an extreme **0.129%** positive rate), standard machine learning practices fail:
- **Accuracy is useless:** A naive majority classifier achieves 99.87% accuracy while catching zero fraudulent transactions.
- **ROC-AUC is misleading:** Because there are over 6.3 million true negatives, a model can generate hundreds of false alarms while maintaining an ROC-AUC above 0.99, because the false positive rate denominator remains vast.

### My Mitigation Strategy:
1. **Metric Prioritization:** I strictly prioritized **Precision-Recall AUC (PR-AUC / Average Precision)**. PR-AUC evaluates the trade-off strictly within positive predictions, directly reflecting an analyst's operational workload.
2. **Loss Weight Tuning:** In XGBoost, I tuned `scale_pos_weight` to `35.94`. This penalized false negatives heavily without causing extreme probability distortion.
3. **Calibrated Action Tiers:** Instead of relying on arbitrary 0.50 binary classification thresholds, I implemented a 4-tier risk routing system:
   - `CRITICAL` ($\ge 0.80$): Automated transaction freeze.
   - `HIGH` ($0.65 - 0.80$): Priority human analyst triage queue.
   - `MEDIUM` ($0.20 - 0.65$): Step-up two-factor authentication challenge.
   - `LOW` ($< 0.20$): Seamless authorization.

---

## 4. Temporal Splitting & Leakage Prevention

One of the most critical engineering disciplines I enforced was strict temporal partitioning. In real banking, payment engines cannot look into the future.

### Why Random Splitting is Prohibited:
Using random $k$-fold cross-validation or `train_test_split(shuffle=True)` on financial transaction data is a catastrophic error. It allows future transactions and settled account balances to leak into the training partition, creating artificial benchmark performance that collapses upon live deployment. Furthermore, in graph modeling, random splitting leaks future edges into historical node embeddings.

### Chronological Cutoffs Implemented:
I split the 743 hours of PaySim chronologically:
- **Training:** Hours 1 – 500 (`4,286,839` records, `5,481` frauds)
- **Validation:** Hours 501 – 620 (`1,029,910` records, `1,366` frauds)
- **Holdout Test:** Hours 621 – 743 (`90,829` records, `1,366` frauds)

### Pre-Authorization Compliance:
I strictly purged post-settlement features (`newbalanceOrig` and `newbalanceDest`) and labels (`isFraud`, `isFlaggedFraud`) from the feature engineering pipeline. Features were engineered exclusively using state available at the millisecond of authorization (e.g. `amount_to_oldbalance_ratio`, `orig_balance_depletion`, `orig_deficit`).

---

## 5. Critical Reflection: Synthetic Dataset Artifacts

A central realization of this project was maintaining methodological skepticism toward the PaySim dataset.

PaySim is an agent-based simulation generated by the MobileMoney simulator. In our holdout test evaluation, XGBoost achieved an extraordinary **PR-AUC of 0.9993** and **ROC-AUC of 0.9994**, catching 1,365 of 1,366 frauds with only 1 false positive.

Rather than taking this near-perfect result at face value, I audited the feature importances and discovered that PaySim's fraud generator exhibits deterministic balance liquidation artifacts:
- In PaySim, fraudulent transfers almost always drain the sender account completely (`amount == oldbalanceOrg`).
- In real-world banking, balance errors occur routinely due to pending holds, merchant processing delays, foreign exchange spreads, and overdraft facilities. Real fraud models rarely achieve PR-AUC above 0.85.

Recognizing this artifact reinforced the necessity of the GraphSAGE model: when fraudsters deliberately vary transfer amounts to bypass tabular balance rules, the relational graph network continues to detect the transaction through topological fan-out and mule clustering.

---

## 6. Graph Memory Scaling & System Constraints

Implementing GraphSAGE on a local workstation required careful memory engineering:
- The PaySim transaction graph contains **763,788 account nodes** and **415,395 settled edges**.
- Storing full adjacency lists and computing multi-hop convolutions on CPU initially risked out-of-memory errors.
- I structured `SentinelGraphSAGE` as an inductive 2-layer network with a historical cutoff step (step 620), freezing the graph topology into an in-memory snapshot. This kept steady-state memory RSS at **~390MB**, allowing smooth execution inside constrained cloud containers (Render Free 512MB RAM limit).

### Limitation Identified:
An in-memory graph snapshot works for hundreds of thousands of nodes, but does not scale to tens of millions of accounts. In a production enterprise deployment, the graph topology must be offloaded to a dedicated distributed graph database such as Neo4j, TigerGraph, or Amazon Neptune.

---

## 7. MLOps, Feast & Kubernetes Engineering Decisions

### 7.1 Feast Feature Store Isolation
In Phase 3, I implemented a working Feast feature store (`feature_store/`) with 5 rolling 24-hour behavioral features.

A key engineering decision was **deliberately isolating Feast from the live production inference path**. Because the deployed XGBoost model was trained on 20 features, injecting Feast's 5 additional features without retraining would have caused schema mismatch and crashed the live Render API. I delivered a verified Feast demonstration with automated point-in-time join tests, and clearly documented the remaining retraining integration.

### 7.2 SQLite Single-Replica Constraint in Kubernetes
In Phase 4, I authored Kubernetes manifests in `k8s/`. I made the conscious decision to configure the backend deployment with `replicas: 1` and `strategy: Recreate`.

I explicitly rejected the temptation to configure multi-replica scaling with SQLite. SQLite uses file-level locking; having multiple pods write to the same PVC simultaneously causes database locking crashes (`sqlite3.OperationalError: database is locked`). A single-replica SQLite setup is sufficient for a prototype, but cloud production requires a managed multi-writer database like PostgreSQL.

---

## 8. Testing & Validation Rigor

Throughout the project, I prioritized automated regression testing over manual verification:
- **Phase 1 (Drift Monitoring):** 6 tests validating Kolmogorov-Smirnov detection, report generation, and safety gates (`ml/tests/test_drift_monitoring.py`).
- **Phase 2 (MLflow Tracking):** 3 tests validating SQLite tracking, parameter/metric logging, and network failure resilience (`ml/tests/test_mlflow_tracking.py`).
- **Phase 3 (Feast Feature Store):** 7 tests validating YAML syntax, offline point-in-time joins, temporal leakage prevention, and SQLite materialization (`ml/tests/test_feast_feature_store.py`).
- **Phase 4 (Kubernetes Manifests):** 9 tests validating manifest syntax, resource limits, health probe configurations, and PVC access modes (`ml/tests/test_kubernetes_manifests.py`).
- **Phase-Specific Verification:** All 25 automated tests across these four newly added phase modules execute cleanly in under 90 seconds.
- **Comprehensive Regression Suite:** In addition to the phase-specific tests, the repository's full Python regression suite passed 145 tests across 14 modules (including EDA, temporal splits, XGBoost, GraphSAGE, hybrid scoring, FastAPI endpoints, transaction ledger, replay, and streaming), all passing with a 100% success rate.

---

## 9. Future Roadmap & Production Improvements

If extending this project further into enterprise production, my priorities would be:
1. **Managed Relational Storage:** Replace the SQLite ledger with managed PostgreSQL on AWS RDS or Cloud SQL to enable multi-replica horizontal scaling.
2. **Distributed Graph Infrastructure:** Migrate the in-memory NetworkX/PyG graph snapshot to Neo4j or Amazon Neptune, utilizing mini-batch subgraph sampling for multi-million account networks.
3. **Live Kafka Verification:** Spin up a managed Kafka broker on Confluent Cloud or AWS MSK to test real-time streaming with live multi-partition consumers.
4. **End-to-End Feast Retraining:** Retrain the tabular model on the 25 combined features (20 pre-auth + 5 Feast rolling aggregates) and promote Feast online retrieval into `inference_service.py`.

---

## 10. Conclusion

The Nexvia Sentinel project was an intensive, rewarding exercise in building modern AI-driven financial infrastructure. Beyond training high-performing models, it demanded strict adherence to temporal causality, methodological skepticism toward synthetic data, careful memory management, and honest engineering documentation.

The resulting system is robust, thoroughly tested, and running live in production today.
