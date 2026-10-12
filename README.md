# Nexvia Sentinel — Real-Time Financial Fraud Detection Platform

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-eb5424.svg)](https://xgboost.ai/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.5+-ee4c2c.svg?logo=pytorch)](https://pytorch-geometric.readthedocs.io/)
[![Feast](https://img.shields.io/badge/Feast-0.66+-3256A4.svg)](https://feast.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11+-0194E2.svg?logo=mlflow)](https://mlflow.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently_AI-0.4+-FF4B4B.svg)](https://evidentlyai.com/)
[![React 18](https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react)](https://react.dev/)

**Nexvia Sentinel** (AIML-01) is an enterprise-grade financial fraud detection platform that combines high-precision tabular gradient boosting (**XGBoost**) with multi-hop account graph neural networks (**PyTorch Geometric GraphSAGE**) to catch both transaction-level anomalies and syndicated money-mule rings in real time.

---

## 🌐 Live Deployments & Authoritative URLs

- **Live Production Frontend (Vercel):** [https://nexvia-sentinel.vercel.app](https://nexvia-sentinel.vercel.app)
- **Live Production Backend (Render):** [https://nexvia-sentinel.onrender.com](https://nexvia-sentinel.onrender.com)
- **Interactive OpenAPI / Swagger Documentation:** [https://nexvia-sentinel.onrender.com/docs](https://nexvia-sentinel.onrender.com/docs)
- **Authoritative Health & Diagnostics Endpoint:** [https://nexvia-sentinel.onrender.com/api/v1/health](https://nexvia-sentinel.onrender.com/api/v1/health)
- **Live Deployment Link:** https://nexvia-sentinel-ebon.vercel.app/

---

## 🏛️ System Architecture

Sentinel combines tabular pre-authorization indicators, relational graph embeddings, drift tracking, and an isolated Feast feature store demonstration.

Detailed diagrams and subsystem breakdowns are available in [docs/architecture.md](docs/architecture.md).

```text
[ Client / Analyst ] ---> [ Vercel React UI ] ---> [ Render FastAPI Backend ]
                                                             |
                 +-------------------------------------------+
                 |
                 +---> [ SentinelFeaturePipeline ] (20 Pre-Auth Features)
                 |            |
                 |            +---> [ XGBoost Model ] (PR-AUC: 0.9993)
                 |            |
                 |            +---> [ GraphSAGE GNN ] (763k nodes, 415k edges)
                 |                         |
                 |                         v
                 +---> [ Hybrid Ensemble & Platt Calibrators ]
                              |
                              v
                 [ 4-Tier Decision Engine ] ---> [ SQLite Ledger DB ]
                              |
                 +------------+------------+
                 |                         |
                 v                         v
     [ Evidently Drift Monitor ]     [ MLflow Tracking ]
     (Reports & Retrain Gate)        (sqlite:///mlruns.db)

* Sandboxed Extensions:
  - Feast Feature Store (feature_store/): Offline Parquet & Online SQLite (Verified in sandbox; isolated from live inference)
  - Kubernetes Manifests (k8s/): Validated statically (Deployment unverified on host)
  - Kafka Streaming Backbone: Tested with in-memory mocks (Real broker unverified)
```

---

## 📋 Comprehensive Implementation & Verification Matrix

| Component / Requirement | Implementation State | Verification Evidence & Test Coverage |
| :--- | :--- | :--- |
| **PaySim EDA & Temporal Splitting** | **Implemented & Tested** | Non-overlapping chronological partitions (Train 1–500, Val 501–620, Test 621–743). Strict pre-auth cutoffs (`ml/pipelines/temporal_splitter.py`). |
| **Tabular XGBoost Baseline** | **Implemented & Deployed** | `scale_pos_weight=35.94`. Holdout Test PR-AUC: `0.9993`, ROC-AUC: `0.9994` on 90,829 transactions. Single-transaction latency: `12.3 ms`. |
| **GraphSAGE Inductive GNN** | **Implemented & Deployed** | 2-layer GraphSAGE with edge classification head. In-memory snapshot: `763,788` nodes, `415,395` edges. Test ROC-AUC: `0.9039`, PR-AUC: `0.4448`. |
| **Hybrid Risk Scoring & Calibration** | **Implemented & Deployed** | Composite scoring: $\alpha P_{XGB} + (1-\alpha) P_{GNN}$ ($\alpha=0.78$). Platt calibrated, 4 tiers (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`). |
| **FastAPI Backend & Persistence** | **Implemented & Deployed** | Production REST API with `/predict/fraud`, `/alerts`, `/replay`, `/health`. Persistent SQLite ledger with analyst feedback loops. |
| **React Analyst Dashboard** | **Implemented & Deployed** | Vite + React 18 UI with real-time prediction, alert status mutation, replay execution, and responsive analytics. |
| **Replay Monitoring with Test Dataset**| **Implemented & Deployed** | Packaged `ml/data/processed/test.parquet` in production Docker image. Fully functional in Render Free deployment. |
| **MLflow Experiment Tracking** | **Implemented & Tested** | Pinned to `sqlite:///mlruns.db`. Logs parameters, metrics (`val_pr_auc`, `val_roc_auc`), and model artifacts. Verified via `test_mlflow_tracking.py`. |
| **Evidently Drift Monitoring** | **Implemented & Tested** | Kolmogorov-Smirnov statistical tests generating HTML/JSON reports (`monitoring_reports/`). Includes safety-gated retraining trigger (`test_drift_monitoring.py`). |
| **Feast Feature Store** | **Implemented in Isolated Sandbox** | Valid `feature_store.yaml`, entity `account_id`, rolling view `account_hourly_stats`. Point-in-time join and SQLite materialization verified in sandbox (`test_feast_feature_store.py`). **Isolated from live inference**. |
| **Kubernetes / Minikube Support** | **Statically Validated Only** | 7 manifests in `k8s/` (Deployment, Service, PVC, ConfigMap). Sizing and probes validated via `test_kubernetes_manifests.py`. **Cluster deployment unverified** due to lack of local Docker/Minikube. |
| **Apache Kafka Streaming Engine** | **Mock-Tested Only** | Worker and producer logic implemented in `backend/app/workers/kafka_consumer.py`. **Real-broker end-to-end streaming unverified**. |

---

## ⚠️ Critical Synthetic Dataset & Architecture Disclosures

### 1. PaySim Dataset Artifacts & Inflated Metrics
- **Synthetic Origin**: PaySim is an agent-based simulation derived from Mobile Money logs in an African country.
- **Inflated Metrics Disclosure**: The exceptionally high XGBoost performance (**PR-AUC: 0.9993**, **ROC-AUC: 0.9994**) is partly driven by deterministic balance simulation artifacts. In PaySim, fraudulent transfers frequently deplete an account completely, causing exact arithmetic patterns such as `amount == oldbalanceOrg` and `newbalanceOrig == 0`.
- **Real-World Reality**: In real banking systems, balance errors arise from pending holds, overdraft fees, multi-currency conversions, and delayed settlement. Real-world fraud models typically achieve PR-AUC in the `0.60 - 0.85` range.
- **Value of GraphSAGE**: The relational GraphSAGE model evaluates topological behavioral anomalies (fan-out disbursement, mule rings, high-degree hubs) independently of balance fields, providing essential protection when fraudsters avoid balance liquidation patterns.

### 2. Single-Replica SQLite Persistence
- **Constraint**: The audit ledger (`sentinel_ledger.db`) utilizes SQLite. SQLite relies on operating system file locks.
- **Limitation**: Mounting SQLite in a multi-pod cluster will cause `sqlite3.OperationalError: database is locked`. The Kubernetes deployment is deliberately constrained to `replicas: 1`. In high-throughput cloud environments, SQLite must be replaced with PostgreSQL (e.g., AWS RDS or Cloud SQL).

### 3. In-Memory Graph Memory Scaling
- **Constraint**: PyTorch Geometric maintains account node embeddings and relational edges in host memory.
- **Limitation**: The current PaySim snapshot consumes ~390MB for 763k nodes. At 100M+ accounts, in-memory representation exceeds single-node RAM budgets and requires a distributed graph database (e.g., Neo4j, Amazon Neptune).

### 4. Feast Feature View Isolation
- **Boundary**: Production inference consumes exactly the 20 pre-auth features from `SentinelFeaturePipeline`.
- **Sandbox**: The Feast feature store (`account_hourly_stats`) maintains 5 rolling multi-hour features. It is demonstrated and verified in an isolated module (`feature_store/`) to protect production stability until model retraining incorporates the 25 combined features.

---

## 🛠️ Local Setup, Testing & Operational Commands

### 1. Environment Setup (Python 3.10)
```powershell
# Clone and enter directory
cd C:\Users\lalit\OneDrive\Documents\Nexvia\AIML-01

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Launching Local Services
**Backend (FastAPI):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation: `http://127.0.0.1:8000/docs`

**Frontend (React / Vite):**
```powershell
cd frontend
npm install
npm run dev
```
Dashboard: `http://localhost:5173`

---

### 3. MLflow Tracking UI
To inspect logged experiments, hyperparameter runs, PR-AUC curves, and model artifacts:
```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
```
Open `http://localhost:5000` in your browser.

---

### 4. Evidently Drift Monitoring Pipeline
Generate HTML and JSON drift reports comparing validation reference data against serving batches:
```powershell
.\.venv\Scripts\python.exe ml/pipelines/drift_monitor.py
```
Reports are written to `monitoring_reports/data_drift_report.html` and `monitoring_reports/drift_summary.json`.

---

### 5. Feast Feature Store Demonstration
Execute offline point-in-time historical retrieval, temporal leakage checks, and online SQLite materialization:
```powershell
.\.venv\Scripts\python.exe feature_store/generate_source_data.py
.\.venv\Scripts\python.exe feature_store/run_retrieval_demo.py
```
Feature store documentation: [feature_store/README.md](feature_store/README.md).

---

### 6. Automated Test Suite Execution
Run the full automated test suite (25 tests covering Kubernetes, Feast, MLflow, and Evidently):
```powershell
.\.venv\Scripts\python.exe -m pytest ml/tests/test_kubernetes_manifests.py ml/tests/test_feast_feature_store.py ml/tests/test_drift_monitoring.py ml/tests/test_mlflow_tracking.py -v
```

---

## 📑 Comprehensive Documentation Index

- [Platform Architecture & Subsystem Map](docs/architecture.md)
- [10-Minute Presentation Deck & Live Demo Script](docs/presentation_deck.md)
- [Individual Engineering Reflection](docs/individual_reflection.md)
- [Feast Feature Store Guide](feature_store/README.md)
- [Minikube & Kubernetes Deployment Guide](docs/minikube_deployment_guide.md)
- [Render Cloud Deployment Guide](docs/render_deployment_guide.md)
- [Tabular XGBoost Evaluation Report](docs/xgboost_evaluation_report.md)
- [GraphSAGE Relational GNN Evaluation Report](docs/gnn_evaluation_report.md)
- [Hybrid Risk Combination & Calibration Report](docs/hybrid_evaluation_report.md)
- [Phase 2 Model Audit & Leakage Verification](docs/phase2_model_audit_report.md)
- [PaySim Exploratory Data Analysis Report](docs/eda_report.md)
- [PaySim Data Dictionary](docs/data_dictionary.md)
