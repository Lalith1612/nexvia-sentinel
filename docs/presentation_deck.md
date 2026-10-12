# Nexvia Sentinel — 10-Minute Technical Presentation Deck

**Project:** Nexvia Sentinel (AIML-01)<br/>
**Presenter:** Lalith / Nexvia AI/ML Engineering<br/>
**Total Duration:** 10 Minutes (600 Seconds)<br/>
**Target Audience:** Engineering Leads, Data Science Reviewers, and FinTech Stakeholders

---

## Slide 1: Problem & Objective (0:00 – 1:00)

### Slide Content
- **The Challenge:** Real-time financial fraud detection under sub-50ms SLA.
- **The Vulnerability:** Tabular models catch isolated balance anomalies, but miss coordinated money-mule rings, cyclical laundering, and multi-hop disbursement syndicates.
- **Project Objective:** Build an end-to-end, production-grade fraud detection platform uniting:
  1. High-precision tabular gradient boosting (**XGBoost**).
  2. Multi-hop relational graph neural networks (**PyTorch Geometric GraphSAGE**).
  3. Pre-authorization feature engineering (strictly zero future or post-settlement leakage).
  4. Real-time REST API, durable analyst audit ledger, and live web dashboard.

### Speaker Notes
> "Good morning, everyone. Today I'm presenting Nexvia Sentinel, an end-to-end financial fraud detection platform developed for real-time payment infrastructures.
>
> In digital banking, fraud detection faces two simultaneous challenges: strict real-time latency budgets under 50 milliseconds, and sophisticated syndicates. While classical machine learning models are effective at catching single-transaction anomalies, they fail to see structural patterns—such as rapid fan-out disbursements and money-mule rings.
>
> Our objective was to engineer a dual-engine architecture that blends XGBoost tabular detection with an inductive PyTorch Geometric GraphSAGE network, backed by a production FastAPI service, persistent analyst triage ledger, and automated drift monitoring."

---

## Slide 2: PaySim Dataset & Severe Class Imbalance (1:00 – 2:00)

### Slide Content
- **Dataset:** PaySim Synthetic Financial Dataset (6.36M transactions across 743 temporal steps).
- **Severe Class Imbalance:**
  - Total Transactions: `6,362,620`
  - Total Fraudulent Transactions: `8,213` (Prevalence: **0.129%**, ~1 fraud per 775 transactions).
- **Transaction Types:** Fraud occurs exclusively in `TRANSFER` and `CASH_OUT` flows.
- **Critical Synthetic Disclosure:**
  - PaySim is generated via an agent-based Mobile Money simulator.
  - Deterministic balance depletion artifacts (`amount == oldbalanceOrg`, `newbalanceOrig == 0`) create near-perfect tabular separability not seen in real banking with pending holds and overdrafts.

### Speaker Notes
> "Let's examine the data foundation. We utilized the PaySim dataset, comprising 6.36 million financial transactions across 743 simulated hours.
>
> The central data challenge is extreme class imbalance: only 0.129% of transactions are fraudulent. Furthermore, exploratory data analysis revealed that fraud occurs exclusively within two transaction types: `TRANSFER` and `CASH_OUT`.
>
> It is essential to be methodologically transparent about the data: PaySim is an agent-based simulation. In PaySim, fraudsters frequently drain accounts entirely. This introduces arithmetic balance artifacts where amount exactly equals the originating balance. While this produces high tabular benchmark scores, real-world systems deal with partial holds, currency spreads, and delayed settlement—which is precisely why topological graph analysis is necessary."

---

## Slide 3: Feature Engineering & Temporal Leakage Prevention (2:00 – 3:00)

### Slide Content
- **Chronological, Non-Overlapping Splitting:**
  - **Train:** Steps 1 – 500 (`4,286,839` txns, `5,481` frauds)
  - **Validation:** Steps 501 – 620 (`1,029,910` txns, `1,366` frauds)
  - **Holdout Test:** Steps 621 – 743 (`90,829` txns, `1,366` frauds)
- **Zero Temporal Leakage:** Strict chronological cutoffs prevent future information from contaminating historical features.
- **Pre-Authorization Compliance:**
  - Excluded post-settlement attributes: `newbalanceOrig`, `newbalanceDest`.
  - Excluded target variables: `isFraud`, `isFlaggedFraud`.
  - Engineered 20 instantaneous pre-auth features (`amount_to_oldbalance_ratio`, `orig_balance_depletion`, `orig_deficit`, `hour_of_day`, `day_of_week`, one-hot types).

### Speaker Notes
> "Feature engineering in fraud detection requires strict temporal discipline. We enforced a chronological split: steps 1 to 500 for training, 501 to 620 for validation, and 621 to 743 as an untouched holdout test partition. Random k-fold cross-validation was strictly prohibited, as it leaks future graph edges and balance distributions into past scoring.
>
> Crucially, our feature pipeline operates under pre-authorization constraints. Features must only utilize information available at the exact millisecond the transaction arrives. We completely excluded post-settlement fields like `newbalanceOrig` and `newbalanceDest`, engineering 20 instantaneous risk features including balance depletion indicators and transaction velocity."

---

## Slide 4: XGBoost Baseline & Precision-Recall Evaluation (3:00 – 4:00)

### Slide Content
- **Model:** XGBoost Classifier (`tree_method='hist'`, `n_estimators=100`, `max_depth=6`).
- **Imbalance Compensation:** `scale_pos_weight = 35.94` (tuned on validation set).
- **Holdout Test Set Results (90,829 untouched transactions):**
  - **PR-AUC (Average Precision):** `0.9993`
  - **ROC-AUC:** `0.9994`
  - **Precision @ 0.620 Threshold:** `99.93%` (1,365 TP, 1 FP)
  - **Recall @ 0.620 Threshold:** `99.93%` (1,365 TP, 1 FN)
- **Serving Performance:** Single-transaction inference latency is `12.3 ms` (`0.0032 ms/txn` in batch).

### Speaker Notes
> "For our tabular baseline, we trained an XGBoost model using histogram-based gradient boosting and class-weight rebalancing.
>
> Because of severe class imbalance, ROC-AUC is misleading because false positives are easily masked by millions of true negatives. We prioritized Precision-Recall AUC (Average Precision). On our holdout test set of 90,829 transactions, XGBoost achieved a PR-AUC of 0.9993 and an ROC-AUC of 0.9994, capturing 1,365 of the 1,366 fraudulent transactions with only a single false positive.
>
> Furthermore, inference latency averaged 12.3 milliseconds on CPU, comfortably satisfying sub-50 millisecond payment authorization SLAs."

---

## Slide 5: GraphSAGE Inductive GNN & Hybrid Risk Scoring (4:00 – 5:00)

### Slide Content
- **Account Relational Graph:**
  - Directed multigraph: accounts as nodes, transactions as temporal edges.
  - Snapshot: `763,788` nodes, `415,395` settled edges.
- **Model Architecture:** 2-layer inductive `SentinelGraphSAGE` with edge classification head.
  - Node embeddings generated via neighbor feature aggregation.
  - Holdout Test ROC-AUC: `0.9039`, PR-AUC: `0.4448` (30x improvement over random 0.015 baseline).
- **Hybrid Risk Combination:**
  $$Risk_{composite} = 0.78 \cdot P_{XGB} + 0.22 \cdot P_{GNN}$$
- **Platt Scaling:** Calibrates margin scores into true posterior probabilities.
- **4-Tier Action Routing:** `CRITICAL` ($\ge 0.80$), `HIGH` ($0.65 - 0.80$), `MEDIUM` ($0.20 - 0.65$), `LOW` ($< 0.20$).

### Speaker Notes
> "While XGBoost catches arithmetic anomalies, syndicated fraudsters deliberately keep individual transaction amounts below threshold limits. To detect these syndicates, we built a relational graph network using PyTorch Geometric.
>
> We modeled accounts as nodes and transactions as directed edges, seeding an inductive 2-layer GraphSAGE network over a graph of 763,000 accounts. GraphSAGE aggregates topological signals—such as burst fan-outs and cyclical hops. On the holdout test set, GraphSAGE achieved an ROC-AUC of 0.9039 and a PR-AUC of 0.4448, representing a 30-fold lift over random prevalence.
>
> We combined both models into a hybrid ensemble using Platt scaling calibration: 78% weight on XGBoost and 22% on GraphSAGE, routing scores into four actionable tiers from low-risk approval to automated critical freezes."

---

## Slide 6: FastAPI Backend & React Analyst Dashboard (5:00 – 6:00)

### Slide Content
- **FastAPI Production Engine:**
  - Startup lifespan pre-loads all models and graph snapshot once into RAM.
  - Sub-50ms inference endpoints: `/api/v1/predict/fraud`, `/api/v1/health`, `/api/v1/replay`.
- **Durable Audit Ledger:** SQLite database (`sentinel_ledger.db`) storing transaction scores, risk tiers, and analyst feedback triage history.
- **Interactive Analyst Dashboard (React 18 + Vite):**
  - Real-time transaction risk scoring with JSON payload editor.
  - Fraud alert queue with interactive triage (`CONFIRMED_FRAUD`, `FALSE_POSITIVE`).
  - Historical replay simulator executing batches from packaged `test.parquet`.
- **Live Deployments:** Live backend on Render Free; live UI on Vercel.

### Speaker Notes
> "To deliver these models to end users, we built a production-grade FastAPI service and a modern React dashboard.
>
> The FastAPI service initializes model weights and graph state once at startup, guaranteeing zero model reloading in the request path. An embedded SQLite ledger tracks every scored transaction and provides analyst triage feedback loops for ongoing auditability.
>
> The frontend provides real-time transaction scoring, alert triage workflows, and an execution replay monitor that runs historical evaluation batches against the holdout dataset. Both are deployed live today on Render and Vercel."

---

## Slide 7: MLflow Tracking & Evidently Drift Monitoring (6:00 – 7:00)

### Slide Content
- **MLflow Experiment Tracking:**
  - Local tracking backend (`sqlite:///mlruns.db`).
  - Logs hyperparameter configurations, dataset split hashes, PR-AUC, ROC-AUC, and serialized model artifacts.
  - Interactive UI launched via `mlflow ui --backend-store-uri sqlite:///mlruns.db`.
- **Evidently AI Drift Monitoring:**
  - Compares reference validation distributions against real-time serving batches.
  - Uses two-sample Kolmogorov-Smirnov tests across continuous features (`amount`, balance depletion).
  - Automatically exports HTML and JSON audit reports (`monitoring_reports/`).
  - **Human-Gated Retraining Trigger:** Triggers recommendations when drift score exceeds 0.30, with strict programmatic gates preventing unauthorized auto-execution.

### Speaker Notes
> "MLOps and monitoring are vital for production fraud systems.
>
> For experiment tracking, we integrated MLflow backed by a local SQLite registry. Every training run tracks hyperparameters, split metadata, evaluation metrics, and serialized model artifacts, accessible through the local MLflow dashboard.
>
> For distribution stability, we implemented Evidently AI. The drift pipeline runs two-sample Kolmogorov-Smirnov statistical tests comparing validation reference baselines against serving batches. If drift exceeds our threshold of 0.30, the system generates an audit alert and recommends model retraining. Crucially, this retraining trigger is safely gated: it requires analyst authorization and strictly prevents unverified automated model replacement."

---

## Slide 8: Feast Feature Store Implementation (7:00 – 8:00)

### Slide Content
- **Feast Repository (`feature_store/`):**
  - Feast version: `0.66.0`.
  - Local configuration: Parquet file offline store and SQLite online key-value store (`online_store.db`).
- **Entity & Features:**
  - Entity: `account_id`
  - Feature View: `account_hourly_stats` (5 rolling 24h metrics: avg amount, max amount, txn count, zero balance frequency, merchant ratio).
- **Verification:**
  - Offline point-in-time joins strictly prevent temporal lookahead bias.
  - Online materialization enables sub-millisecond key-value retrieval.
- **Production Isolation Boundary:**
  - Demonstrated in an isolated module; **not wired into live production inference** to avoid breaking the 20-feature input schema of existing deployed models.

### Speaker Notes
> "In Phase 3, we implemented an isolated Feast feature store using Feast 0.66.
>
> We defined an `account_id` entity and an `account_hourly_stats` feature view containing five rolling 24-hour behavioral metrics. Feast uses a local Parquet offline store and a SQLite online store.
>
> Our automated tests verify that historical point-in-time joins strictly prevent future data leakage: queries before an event timestamp return no future data, while online materialization enables rapid key-value retrieval.
>
> We maintained strict engineering honesty: Feast is demonstrated in an isolated module. Because our active production models are trained on 20 features, injecting Feast's 5 rolling features into production inference would cause input schema mismatch. We verified Feast in sandbox and documented the remaining retraining integration."

---

## Slide 9: Kubernetes Architecture & Verification Status (8:00 – 9:00)

### Slide Content
- **Kubernetes Architecture (`k8s/`):**
  - Pod topology: `sentinel-backend` (FastAPI) and `sentinel-frontend` (Nginx Alpine).
  - Reverse-proxy routing: Frontend Nginx reverse-proxies `/api/` to `sentinel-backend:8000`.
  - Storage: `sentinel-backend-pvc` (1Gi, `ReadWriteOnce`).
  - Probes: Readiness and liveness probes configured on `/api/v1/health` and `/`.
  - Sizing: Backend requests `512Mi` RAM, limits `1536Mi` (headroom for GraphSAGE).
- **Verification Status Disclosure:**
  - 7 manifests statically validated via automated test suite `test_kubernetes_manifests.py` (9/9 passed).
  - **Cluster deployment is UNVERIFIED**: Docker and Minikube are not installed on this local Windows machine. Documented in `docs/minikube_deployment_guide.md`.

### Speaker Notes
> "In Phase 4, we prepared containerization and Kubernetes manifests for local cluster deployment.
>
> We created manifests for the backend, frontend, ConfigMap, and PersistentVolumeClaim. The frontend is containerized using a multi-stage Dockerfile that serves compiled assets via Nginx and proxies `/api/` calls directly to the internal backend service. Sizing accounts for PyTorch memory, requesting 512MB and limiting to 1.5GB RAM.
>
> Here again, we maintain strict transparency: all seven manifests passed automated YAML syntax and structural tests. However, because Docker and Minikube are not installed on this local Windows host, we explicitly state that live cluster deployment has not been run on this machine, and we have fully documented the execution steps in our Minikube deployment guide."

---

## Slide 10: Limitations, Lessons & Conclusion (9:00 – 10:00)

### Slide Content
- **Current Limitations:**
  - *Single-Replica SQLite:* File-level locking requires `replicas: 1`; needs PostgreSQL for multi-pod scaling.
  - *In-Memory Graph:* 763k nodes consume ~390MB RAM; 100M+ accounts requires a distributed graph database (e.g. Neo4j).
  - *Kafka Real-Broker:* Tested with in-memory mocks; live cluster streaming requires cloud broker validation.
- **Key Engineering Lessons:**
  - Synthetic datasets require skepticism: tabular metrics can be inflated by balance simulation quirks.
  - Strict pre-authorization feature engineering is essential to prevent costly real-world leakage.
  - Isolated staging allows validating modern tools (Feast, K8s) without disrupting working cloud production.
- **Summary:** Nexvia Sentinel delivers a verified, hybrid fraud detection engine running live today.

### Speaker Notes
> "To conclude, let's address limitations and future work.
>
> First, SQLite file locking constrains backend scaling to a single replica; cloud production requires managed PostgreSQL. Second, our in-memory graph snapshot is bounded by RAM; scaling beyond 10 million accounts requires a distributed graph store. Third, our Kafka pipeline is validated with unit mocks, but awaits live broker verification.
>
> The major engineering takeaway of this project is the importance of methodological rigor: synthetic datasets like PaySim can inflate tabular metrics, making hybrid graph analysis essential for real-world resilience.
>
> Nexvia Sentinel is live, tested across 25 automated unit and regression tests, and demonstrates a modern, production-grade fraud detection platform. Thank you, and I look forward to your questions."

---

## 🎥 Live Demo Sequence Script (3 Minutes Optional Add-On)

| Step | Action | Endpoint / Screen | Expected Result |
| :--- | :--- | :--- | :--- |
| **1. Service Health** | Open `https://nexvia-sentinel.onrender.com/api/v1/health` | Browser JSON view | Shows `status: "HEALTHY"`, `artifacts_loaded: true`, `total_indexed_nodes: 763788`. |
| **2. High-Risk Scoring** | Navigate to `https://nexvia-sentinel.vercel.app/predict`, enter high amount `TRANSFER` draining balance | Predict Screen | Composite risk $> 0.85$, decision badge shows **CRITICAL - REJECT & FREEZE**. |
| **3. Alert Triage** | Navigate to Alert Queue, click alert, select `CONFIRMED_FRAUD` | Alerts Screen | Alert status mutates to `RESOLVED`, audit trail recorded in SQLite ledger. |
| **4. Historical Replay** | Trigger replay job of 100 records from `test.parquet` | Replay Screen | Progress bar runs to 100%, charts render fraud detection rate and latency distribution. |
| **5. MLflow UI (Local)** | Run `mlflow ui --backend-store-uri sqlite:///mlruns.db` | `localhost:5000` | Displays logged runs, hyperparameters (`scale_pos_weight`), and PR-AUC metrics. |
