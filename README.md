# Sentinel — Real-Time Financial Fraud Detection Platform

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-eb5424.svg)](https://xgboost.ai/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.5+-ee4c2c.svg?logo=pytorch)](https://pytorch-geometric.readthedocs.io/)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-3.6+-231F20.svg?logo=apachekafka)](https://kafka.apache.org/)
[![Feast](https://img.shields.io/badge/Feast-0.37+-3256A4.svg)](https://feast.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11+-0194E2.svg?logo=mlflow)](https://mlflow.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently_AI-0.4+-FF4B4B.svg)](https://evidentlyai.com/)
[![React 18](https://img.shields.io/badge/React-18+-61DAFB.svg?logo=react)](https://react.dev/)

**Sentinel** is an enterprise-grade, real-time financial fraud detection platform designed to detect both individual transactional anomalies and syndicated fraud rings.

Sentinel powers end-to-end detection by blending high-precision tabular gradient boosting (**XGBoost**) with multi-hop account graph neural networks (**PyTorch Geometric**), fed by an event streaming backbone (**Apache Kafka**) and low-latency feature store (**Feast**).

---

## 🏛️ System Architecture

```text
                               ┌────────────────────────────────┐
                               │     PaySim Streaming Feed      │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   Apache Kafka Stream Broker   │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   Kafka Consumer / Worker      │
                               └───────┬────────────────┬───────┘
                                       │                │
            ┌──────────────────────────┘                └─────────────────────────┐
            ▼                                                                     ▼
┌───────────────────────┐                                             ┌───────────────────────┐
│  Feast Feature Store  │                                             │ Account Graph Engine  │
│  (Real-Time Balances) │                                             │ (PyTorch Geometric)   │
└───────────┬───────────┘                                             └───────────┬───────────┘
            │                                                                     │
            ▼                                                                     ▼
┌───────────────────────┐                                             ┌───────────────────────┐
│     XGBoost Model     │                                             │  GNN GraphSAGE Model  │
│  (Tabular Anomaly)    │                                             │   (Fraud Ring Mule)   │
└───────────┬───────────┘                                             └───────────┬───────────┘
            │                                                                     │
            └──────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
                       ┌────────────────────────────────┐
                       │ Hybrid Risk Ensemble & Scoring │
                       └───────────────┬────────────────┘
                                       │
                       ┌───────────────┴────────────────┐
                       ▼                                ▼
            ┌─────────────────────┐          ┌───────────────────────┐
            │  FastAPI & Database │          │ WebSocket Live Stream │
            │ (Alerts, Cases, DB) │          │ (Sentinel React UI)   │
            └──────────┬──────────┘          └───────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │ Evidently & MLflow  │
            │ (Drift & Monitoring)│
            └─────────────────────┘
```

---

## 📁 Repository Directory Structure

```text
AIML-01/
├── .env.example              # Environment variables template
├── .gitignore                # Comprehensive ignore rules
├── pyproject.toml            # Project dependencies & build config
├── README.md                 # Project documentation & run guides
├── backend/                  # FastAPI REST & WebSocket backend
│   ├── app/                  # Application code (API, Core, Models, Services, Workers)
│   ├── tests/                # Backend API & service tests
│   ├── requirements.txt      # Backend Python dependencies
│   └── Dockerfile            # Container definition for backend
├── ml/                       # Machine Learning & Graph pipelines
│   ├── data/                 # Raw, processed, and sample datasets
│   ├── features/             # Feature extraction and engineering logic
│   ├── models/               # Serialized models and artifact weights
│   ├── notebooks/            # Exploratory analysis & prototyping
│   ├── pipelines/            # Training, evaluation & drift scripts
│   └── tests/                # Model evaluation and leakage tests
├── feature_store/            # Feast Feature Store definitions
├── frontend/                 # Sentinel React + TypeScript + Vite UI
├── k8s/                      # Kubernetes deployment & service manifests
├── docker-compose.yml        # Orchestration for multi-container stack
└── docs/                     # Technical specifications & submission notes
```

---

## 🚀 Execution & Verification Roadmap

- **Phase 1:** Dataset Inspection, Memory-Safe Preprocessing & Temporal Leakage-Free Splitting
- **Phase 2:** Feature Engineering & Tabular XGBoost Baseline with PR-AUC / Latency Benchmarks
- **Phase 3:** Account Graph Network Construction & PyTorch Geometric GNN Fraud-Ring Model
- **Phase 4:** Hybrid Risk Combination Strategy, Explainability & Score Calibration
- **Phase 5:** MLflow Experiment Tracking & Model Registry Integration
- **Phase 6:** Feast Feature Store Setup & Real-Time Online Retrieval
- **Phase 7:** Apache Kafka Transaction Streaming Producer & Consumer Pipeline
- **Phase 8:** Production FastAPI Backend, Persistence & Analyst Feedback Loops
- **Phase 9:** Sentinel Fintech Enterprise UI (React, Vite, Cytoscape Graph Explorer)
- **Phase 10:** Docker Compose, Kubernetes/Minikube Manifests & Internship Submission Package
