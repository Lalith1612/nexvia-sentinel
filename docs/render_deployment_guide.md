# Nexvia Sentinel — Render Cloud Deployment Guide

This guide details how to deploy the **Nexvia Sentinel** FastAPI backend as a Docker-based Web Service on [Render](https://render.com), configure persistent SQLite storage, and connect the deployed Vercel frontend.

---

## 1. Architecture & Deployment Prerequisites

- **GitHub Repository**: `https://github.com/Lalith1612/nexvia-sentinel` (Private)
- **Deployment Type**: Docker Web Service
- **Runtime Environment**: Linux x86_64, Python 3.10 slim, CPU-only PyTorch
- **Concurrency Mode**: **Strictly 1 Worker** (`--workers 1`) to preserve in-memory relational GraphSAGE state and atomic SQLite WAL transactions.
- **Persistent Disk**: Recommended for the SQLite ledger database to retain alerts, triage history, and settlement ledger state across redeploys.

---

## 2. Render Web Service Configuration

1. In the **Render Dashboard**, click **New +** and select **Web Service**.
2. Select your repository: **`Lalith1612/nexvia-sentinel`**.
3. Configure the service parameters:
   - **Name**: `nexvia-sentinel-api`
   - **Region**: Select your preferred region (e.g., *Oregon (US West)* or *Frankfurt (EU Central)*)
   - **Branch**: `master`
   - **Runtime**: **Docker** (Render will automatically detect the root [`Dockerfile`](file:///c:/Users/lalit/OneDrive/Documents/Nexvia/AIML-01/Dockerfile))
   - **Instance Type**: **Starter** ($7/month, 512 MB – 2 GB RAM recommended). *Note: The cold-start inductive model requires ~300 MB RAM at idle; Starter comfortably avoids OOM thresholds.*

---

## 3. Persistent Disk Configuration (SQLite Ledger)

Render web services operate on an ephemeral filesystem unless a Persistent Disk is attached. To prevent alert and ledger data from resetting on redeploy:

1. In the service settings, scroll to **Disks** and click **Add Disk**.
2. Configure:
   - **Name**: `sentinel-ledger-data`
   - **Mount Path**: `/data`
   - **Size**: `1 GB` (or standard tier minimum)
3. The application will store its SQLite database at `/data/sentinel_ledger.db` with WAL mode enabled.

---

## 4. Environment Variables

In the **Environment** tab of your Render service, configure the following variables:

| Key | Example Value | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `production` | Deployment mode |
| `WORKERS` | `1` | Strict single worker constraint |
| `CORS_ORIGINS` | `https://your-nexvia-sentinel.vercel.app,http://localhost:5173` | Comma-separated or JSON list of authorized frontend domains |
| `LEDGER_DB_PATH` | `/data/sentinel_ledger.db` | Path inside the mounted disk |
| `KAFKA_ENABLED` | `false` | Disabled by default (keeps background consumer worker inactive) |
| `USE_MOCK_STREAMING` | `false` | Explicitly flags that mock streaming is inactive |

*Note: Render automatically injects the `PORT` environment variable (e.g. `10000`). The Dockerfile startup command dynamically reads `${PORT:-8000}`.*

---

## 5. Health Check Configuration

In the service **Advanced Settings**:
- **Health Check Path**: `/api/v1/health`

Render will poll this endpoint before switching traffic to new deployments.

---

## 6. Post-Deployment Verification

Once the build finishes and Render displays **Live**, perform the following verification checks:

### Step 1 — Check Service Health
```bash
curl -s https://your-nexvia-sentinel-api.onrender.com/api/v1/health | jq
```
**Expected Response**:
```json
{
  "status": "HEALTHY",
  "service_name": "Sentinel Real-Time Fraud Detection Platform",
  "version": "0.1.0",
  "artifacts_loaded": true,
  "graph_state_initialized": true,
  "snapshot_cutoff_step": 0,
  "max_settled_step": 0,
  "total_indexed_nodes": 0,
  "settled_edges_count": 0,
  "torch_device": "cpu",
  "memory_rss_mb": 285.4
}
```

### Step 2 — Verify Model Information & Ensemble Weights
```bash
curl -s https://your-nexvia-sentinel-api.onrender.com/api/v1/model/info | jq
```
**Expected Response**:
- Confirms dynamic ensemble weights: `0.78` XGBoost + `0.22` GraphSAGE.
- Displays risk tier boundaries (`CRITICAL >= 0.80`, `HIGH >= 0.65`, `MEDIUM >= 0.20`, `LOW >= 0.00`).
- Confirms Platt probability calibration method and regulatory governance notes.

### Step 3 — Test Live Fraud Prediction (Pre-Authorization)
```bash
curl -X POST https://your-nexvia-sentinel-api.onrender.com/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 625,
    "type": "TRANSFER",
    "amount": 250000.0,
    "nameOrig": "C1234567890",
    "oldbalanceOrg": 250000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C9876543210",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
  }' | jq
```
**Expected Response**:
- Valid `fraud_risk_score` (between `0.0` and `1.0`).
- `risk_tier` (`CRITICAL`, `HIGH`, etc.) and `recommended_action`.
- SHAP feature attribution breakdown and inductive GraphSAGE topological node context.

---

## 7. Connecting the Vercel Frontend

1. Go to your **Vercel Project Dashboard** for `nexvia-sentinel`.
2. Navigate to **Settings** > **Environment Variables**.
3. Set or update:
   ```env
   VITE_API_BASE_URL=https://your-nexvia-sentinel-api.onrender.com
   ```
4. Trigger a frontend redeploy on Vercel so the production bundle points to the live Render backend.
5. In the Render environment settings, ensure `CORS_ORIGINS` includes your live Vercel URL:
   ```env
   CORS_ORIGINS=https://your-nexvia-sentinel.vercel.app,http://localhost:5173
   ```
