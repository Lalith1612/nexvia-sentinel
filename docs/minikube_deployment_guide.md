# Nexvia Sentinel — Minikube & Kubernetes Deployment Guide (Phase 4)

## 1. Overview & Architecture
This guide provides the complete procedure for building, deploying, and verifying the **Nexvia Sentinel AIML-01** fraud detection platform on a local Kubernetes cluster using **Minikube**.

The Kubernetes architecture separates the stateless presentation layer from the stateful ML inference engine:

```
+-----------------------------------------------------------------------------------+
|                              MINIKUBE CLUSTER                                     |
|                                                                                   |
|   +--------------------------+                +-------------------------------+   |
|   |   sentinel-frontend      |                |      sentinel-backend         |   |
|   |   (Vite + React + Nginx) |   ClusterIP    |   (FastAPI + XGBoost + GNN)   |   |
|   |   Port 80 (NodePort:     | -------------> |   Port 8000                   |   |
|   |            30080)        |  /api/ proxy   |   Replicas: 1 (Stateful Graph)|   |
|   +--------------------------+                +---------------+---------------+   |
|                ^                                              |                   |
|                |                                              v                   |
|         Browser / Analyst                            +-----------------+          |
|                                                      |  sentinel-pvc   |          |
|                                                      |  /data (SQLite) |          |
|                                                      +-----------------+          |
+-----------------------------------------------------------------------------------+
```

---

## 2. Resource Sizing & Storage Constraints

### Memory & CPU Allocation
The backend loads several ML artifacts at startup:
- `xgboost_baseline.json`: Gradient-boosted decision tree baseline.
- `gnn_graphsage.pt` & `torch`: 2-layer inductive GraphSAGE network.
- `hybrid_calibrators.joblib`: Platt scaling calibrators.
- Historical graph topology: In-memory networkx/PyG graph structure containing 760k+ nodes and 415k+ edges.

| Component | CPU Request / Limit | Memory Request / Limit | Rationale |
| :--- | :--- | :--- | :--- |
| **Backend** | `250m` / `1000m` | `512Mi` / `1536Mi` | Peak memory RSS reaches ~500MB during startup graph tensor loading, stabilizing at ~30MB idle. |
| **Frontend** | `50m` / `250m` | `64Mi` / `256Mi` | Lightweight Alpine Nginx reverse proxy serving static pre-compiled assets. |

### SQLite Persistence & Single-Replica Limitation
> [!WARNING]
> **Single-Replica & SQLite Persistence**:
> - The backend is configured with `replicas: 1` and `strategy: Recreate`.
> - **In-Memory Graph State**: GraphSAGE transaction history is maintained in-process. Multi-replica scaling without distributed graph synchronization would lead to split-brain node state.
> - **SQLite Locking**: The audit ledger (`/data/sentinel_ledger.db`) uses SQLite mounted via a `ReadWriteOnce` PVC. Multiple pods attempting concurrent writes to a shared SQLite database trigger `sqlite3.OperationalError: database is locked`.
> - **Production Caveat**: A single-replica SQLite configuration does **not** provide high-availability distributed storage. In full cloud production, SQLite should be migrated to Amazon RDS / Google Cloud SQL (PostgreSQL), and graph state should be persisted in Neo4j or Redis Graph.

---

## 3. Host Prerequisites & Availability

To run this deployment, the developer workstation requires:
1. **Docker Desktop** (or Docker Engine on Linux)
2. **Minikube** (`v1.32.0` or higher)
3. **kubectl** (`v1.28.0` or higher)

> [!NOTE]
> **Host Environment Status**:
> As of Phase 4 verification, Docker, Minikube, and kubectl are **not installed** on this local Windows host. Automated tests validate manifest syntax, resource configurations, and probe definitions statically. To deploy on a machine with Minikube installed, follow the steps below.

---

## 4. Step-by-Step Deployment Instructions

### Step 1: Start Minikube Cluster
Allocate at least 4 CPUs and 4GB RAM to accommodate PyTorch and graph loading:
```powershell
minikube start --driver=docker --cpus=4 --memory=4096
```

Verify cluster status:
```powershell
kubectl cluster-info
kubectl get nodes
```

---

### Step 2: Build Container Images
You can build images directly inside the Minikube Docker environment to avoid external container registries:

**A. Configure PowerShell to use Minikube's Docker daemon:**
```powershell
& minikube -p minikube docker-env --shell powershell | Invoke-Expression
```

*(On Linux / macOS bash, use: `eval $(minikube docker-env)`)*

**B. Build the Backend Image:**
From repository root (`C:\Users\lalit\OneDrive\Documents\Nexvia\AIML-01`):
```powershell
docker build -t nexvia-sentinel-backend:latest .
```

**C. Build the Frontend Image:**
```powershell
docker build -t nexvia-sentinel-frontend:latest ./frontend
```

*(Alternatively, if built outside Minikube, load them via:)*
```powershell
minikube image load nexvia-sentinel-backend:latest
minikube image load nexvia-sentinel-frontend:latest
```

---

### Step 3: Deploy Manifests
Deploy the complete Sentinel stack using Kustomize:
```powershell
kubectl apply -k k8s/
```
*(Or individually: `kubectl apply -f k8s/`)*

Expected Output:
```text
configmap/sentinel-config created
persistentvolumeclaim/sentinel-backend-pvc created
deployment.apps/sentinel-backend created
service/sentinel-backend created
deployment.apps/sentinel-frontend created
service/sentinel-frontend created
```

---

### Step 4: Verify Deployment Health

**A. Monitor Pod Initialization:**
```powershell
kubectl get pods -l app.kubernetes.io/name=nexvia-sentinel -w
```
Wait until both `sentinel-backend` and `sentinel-frontend` report `STATUS: Running` and `READY: 1/1`.

**B. Check Backend Startup Logs:**
```powershell
kubectl logs deployment/sentinel-backend -c backend
```
Expected Log Signature:
```text
[INFO] [sentinel.app] Starting Sentinel Fraud Inference Service (v0.1.0)...
[INFO] [sentinel.inference] Loading model artifacts from /app/ml/models...
[INFO] [sentinel.inference] Initializing GraphSAGE state snapshot...
[INFO] [sentinel.app] Sentinel artifacts and graph state snapshot loaded successfully.
[INFO] [uvicorn.error] Application startup complete.
[INFO] [uvicorn.error] Uvicorn running on http://0.0.0.0:8000
```

**C. Inspect Authoritative Health Probe:**
```powershell
kubectl exec deployment/sentinel-backend -c backend -- python -c "
import urllib.request, json
with urllib.request.urlopen('http://localhost:8000/api/v1/health') as r:
    print(json.dumps(json.loads(r.read()), indent=2))
"
```
Expected Response:
```json
{
  "status": "HEALTHY",
  "service_name": "Sentinel Real-Time Fraud Detection Platform",
  "version": "0.1.0",
  "artifacts_loaded": true,
  "graph_state_initialized": true,
  "snapshot_cutoff_step": 620,
  "torch_device": "cpu"
}
```

---

### Step 5: Access the Dashboard & API

**Access the Frontend UI:**
Minikube exposes the NodePort service (`nodePort: 30080`):
```powershell
minikube service sentinel-frontend
```
This automatically opens the React dashboard in your default browser at `http://<minikube-ip>:30080`.

**Port-Forwarding Backend Directly (Optional for Swagger Docs):**
```powershell
kubectl port-forward svc/sentinel-backend 8000:8000
```
Open `http://localhost:8000/docs` in your browser for the OpenAPI interactive Swagger UI.

---

## 5. Teardown & Clean Up
To remove the deployed resources:
```powershell
kubectl delete -k k8s/
```
To delete the persistent volume and clean all stored ledger data:
```powershell
kubectl delete pvc sentinel-backend-pvc
```
To stop the Minikube cluster:
```powershell
minikube stop
```
