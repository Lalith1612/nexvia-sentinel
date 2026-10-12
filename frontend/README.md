# Nexvia Sentinel — Financial Fraud Intelligence Console

The **Nexvia Sentinel Console** is a high-density, portfolio-grade risk operations dashboard built with React, TypeScript, and Vite. Branded with the official **Nexvia Technologies** product identity, it interfaces with the Sentinel FastAPI real-time fraud detection engine to deliver pre-authorization transaction inference, alert triage, bounded transaction replays, and infrastructure observability.

---

## Brand & Product Identity

- **Primary Brand Mark**: Nexvia Technologies official emblem (`/nexvia-logo.png`).
- **Product Designation**: **NEXVIA SENTINEL** &middot; *Fraud Intelligence Console*.
- **Aesthetic**: Midnight Cyber-Fintech Operations Console (`#080D1A` base with electric blue `#1677FF` and cyber cyan `#00D9FF` accents).
- **Attribution**: Restrained Nexvia Technologies attribution in sidebar footer and regulatory governance notes.

---

## Architecture & Technology Stack

- **Framework**: React 18 with TypeScript 5.4
- **Bundler & Dev Server**: Vite 5
- **Styling**: Custom Midnight Cyber-Fintech Design System (Vanilla CSS tokens)
- **Icons**: Lucide React
- **Visualization**: Recharts & custom telemetry components
- **Testing**: Vitest with `@testing-library/react` and `@testing-library/jest-dom`
- **API Client**: Centralized typed HTTP client with `ApiError` mapping

---

## Views & Operational Capabilities

1. **Operations Overview (`/overview`)**
   - Live backend health, memory RSS, PyTorch inference device, and relational graph state (763,788 nodes, 415,395 settled edges, step 620 cutoff).
   - Authoritative ensemble parameters dynamically rendered from backend configuration: **78.0% Tabular XGBoost** + **22.0% Relational GraphSAGE**, Platt scaling calibration, and operational risk tier thresholds (`CRITICAL ≥ 0.80`, `HIGH ≥ 0.65`, `MEDIUM ≥ 0.20`, `LOW ≥ 0.00`).
   - Compact KPI grid and real-time ledger alert posture.
   - Non-interfering dry-run replay execution history preview.

2. **Alert Investigation & Triage (`/alerts`)**
   - High-density alert queue with multi-attribute filtering (Status, Risk Tier, Minimum Score, and text search across Alert ID, Transaction ID, and Accounts).
   - Investigation drawer clearly distinguishing model advisory suggestions from executed analyst decisions.
   - State machine-compliant triage updates (`OPEN` → `INVESTIGATING` → `CONFIRMED_FRAUD` / `FALSE_POSITIVE`) with mandatory analyst ID and resolution notes.
   - Immutability guards preventing modifications to finalized terminal alerts.

3. **Pre-Authorization Fraud Prediction (`/prediction`)**
   - Interactive transaction submission form strictly matching `TransactionRequest`.
   - Real-time input validation preventing lookahead bias (`step >= 621`) and invalid amounts.
   - Quick scenario presets for rapid analyst evaluation (High-Risk Drain, Suspicious Cash-Out, Benign Merchant Payment).
   - Deep-dive explainability: Dual ensemble decomposition, XGBoost SHAP feature attributions, and GraphSAGE topological network context.
   - Explicit regulatory governance disclaimer: Probabilistic advisory outputs do not represent legal fraud confirmation or frozen funds.

4. **Bounded Transaction Replay Monitor (`/replay`)**
   - Dry-run simulation launcher across historical test datasets (`test.parquet`).
   - Controlled progress polling with real-time throughput (records/sec), elapsed time, and ETA.
   - Detailed post-job metrics: Financial exposure calculations (total vs. flagged volume), latency percentiles (min, mean, p50, p95, max), and risk tier distribution.
   - Verification of the non-interference guarantee: Strictly 0 alerts created and 0 settlements applied to the ledger.
   - Job cancellation confirmation guard with HTTP 409 conflict protection on terminal states.

5. **Streaming Infrastructure Status (`/streaming`)**
   - Visual transaction event streaming architecture topology.
   - Configuration specifications: Partitioning strategy by `nameOrig` (preserving causal account ordering) and manual post-settlement offset commit.
   - Honest technical status: **REAL BROKER: NOT VERIFIED / LOCAL MOCK** due to Docker and live Kafka broker unavailability in the local environment.

---

## Getting Started

### Prerequisites

- **Node.js**: `v18.x` or higher (verified on Node `v24.14.1`)
- **npm**: `v9.x` or higher
- **Sentinel FastAPI Backend**: Running locally on `http://127.0.0.1:8000`

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment Variables

Create `.env.local` in `frontend/`:

```bash
cp .env.example .env.local
```

Ensure `.env.local` points to your active backend:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### 3. Launch Development Server

```bash
npm run dev
```

The console will run locally on `http://localhost:5173`.

---

## Testing & Production Build

### Run Unit Tests

Execute the Vitest test suite covering API error handling, overview cards, alert triage workflows, prediction form validation, and replay job cancellation:

```bash
npm test
```

### Build for Production

Compile TypeScript and bundle static assets with Vite:

```bash
npm run build
```

Production output will be generated in `frontend/dist/`.

---

## Deployment Guide (Vercel & Cloud)

### Frontend Deployment (Vercel)

The frontend is fully configured for deployment on Vercel:

1. Connect your repository to Vercel and set the root directory to `frontend`.
2. Configure Build Command: `npm run build`
3. Configure Output Directory: `dist`
4. In Vercel Project Settings > **Environment Variables**, define:
   ```env
   VITE_API_BASE_URL=https://your-sentinel-backend-url.com
   ```

### Backend Deployment

The FastAPI application must be hosted separately in a Python environment (e.g., Render, Railway, AWS ECS, or Fly.io):
- Ensure `CORS_ORIGINS` in the backend configuration includes your Vercel frontend domain (e.g., `https://nexvia-sentinel.vercel.app`).
- Model checkpoints in `ml/models/checkpoints/` must be accessible to the FastAPI process during container startup.

---

## Known Limitations

- **Live Kafka Integration Unverified**: Live streaming requires an active Apache Kafka cluster. Because Docker is unavailable in this development environment, the streaming layer operates against verified mock consumer contracts.
- **Dry-Run Non-Interference**: Replay simulations intentionally do not write to the SQLite ledger or create alerts.
- **Authoritative Model Weights**: Model weights (78% XGBoost / 22% GraphSAGE) are dynamically fetched from `GET /api/v1/model/info` and reflect the Phase 3 validation optimization.
