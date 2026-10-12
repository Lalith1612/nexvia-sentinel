# Nexvia Sentinel — Feast Feature Store (Phase 3)

## 1. Overview and Architecture
This repository directory (`feature_store/`) implements a local, free, production-grade **Feast Feature Store** (Feast `0.66.0`) for financial fraud detection in Nexvia Sentinel AIML-01.

It manages pre-authorization entity behavioral profiles across historical transaction windows, enabling:
- **Historical Point-in-Time Joins**: Offline training dataset generation strictly free from temporal lookahead bias or future information leakage.
- **Low-Latency Online Materialization**: Fast key-value feature retrieval by `account_id` for real-time risk scoring.

```
+-------------------------------------------------------------+
|                 Source Transactions (Parquet)               |
|            account_id, event_timestamp, created_at          |
+-------------------------------------------------------------+
                               |
               (Feast Offline Store: Parquet File)
                               |
             +-----------------+-----------------+
             |                                   |
             v                                   v
  [store.get_historical_features]     [store.materialize(...)]
      (Point-in-Time Join)            (Pushes to SQLite DB)
             |                                   |
             v                                   v
  Model Training Datasets              [store.get_online_features]
  (Zero Temporal Leakage)               (Real-Time Low Latency)
```

---

## 2. Local Configuration & Platform Compatibility

The feature store is configured in [`feature_store/feature_store.yaml`](feature_store.yaml) using native file and SQLite stores:

```yaml
project: sentinel_feature_store
registry: data/registry.db
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 3
```

### Limitations of Local Stores:
1. **Offline Store (`file` / Parquet via PyArrow/DuckDB)**:
   - *Strengths*: Completely free, zero background infrastructure, 100% Windows compatible, fast vectorized scans on small/medium datasets.
   - *Limitations*: Not distributed (cannot query multi-terabyte datasets across a cluster without BigQuery, Snowflake, or Spark). Full column scans required for ad-hoc queries.
2. **Online Store (`sqlite`)**:
   - *Strengths*: Single self-contained file (`data/online_store.db`), zero daemon required, robust ACID transactions, reliable cross-platform execution.
   - *Limitations*: Single-writer locking model. Under high concurrent write traffic (thousands of parallel ingestion workers), SQLite locks will queue. In large-scale distributed deployments, Redis, DynamoDB, or PostgreSQL is used for multi-thousand RPS write throughput.

---

## 3. Entity & Feature Definitions

Defined in [`feature_store/features.py`](features.py):

- **Entity**: `account_id` (Primary string identifier for originators `C...` or recipients `M...`/`C...`).
- **Feature View**: `account_hourly_stats`
  - **Source**: `FileSource` pointing to `data/account_features.parquet`.
  - **Timestamp Field**: `event_timestamp` (UTC).
  - **Created Timestamp**: `created_timestamp` (UTC).
  - **TTL**: 14 days.
  - **Features**:
    1. `rolling_avg_amount_24h` (`Float32`): Rolling average transaction amount over the past 24 hours.
    2. `rolling_txn_count_24h` (`Int64`): Transaction count/frequency in the past 24 hours.
    3. `rolling_max_amount_24h` (`Float32`): Maximum transaction amount in the past 24 hours.
    4. `zero_balance_frequency_24h` (`Float32`): Proportion of past transactions where starting balance was zero.
    5. `merchant_interaction_ratio_24h` (`Float32`): Proportion of transfers routed to merchants in past 24 hours.

### Target Leakage & Post-Settlement Safeguards
To adhere to strict pre-authorization scoring requirements:
- **Excluded**: `isFraud`, `isFlaggedFraud` (supervised labels never permitted in feature store).
- **Excluded**: `newbalanceOrig`, `newbalanceDest` (post-settlement account state).
- All rolling aggregates are computed exclusively using historical transactions completed **prior** to authorization time `t`.

---

## 4. Commands & Workflow

### A. Generate Source Parquet
Generates reproducible feature partitions from local dataset:
```powershell
.\.venv\Scripts\python.exe feature_store/generate_source_data.py
```

### B. Register Feast Entities and Views
Applies definitions to the local registry (`data/registry.db`):
```powershell
.\.venv\Scripts\python.exe -c "
import sys; from pathlib import Path; from feast import FeatureStore
p = Path('feature_store').resolve(); sys.path.insert(0, str(p))
from features import account_id, account_hourly_stats_view
store = FeatureStore(repo_path=str(p))
store.apply([account_id, account_hourly_stats_view])
print('Applied successfully!')
"
```

### C. Run End-to-End Retrieval Demo
Demonstrates historical point-in-time retrieval, leakage prevention, materialization, and online lookup:
```powershell
.\.venv\Scripts\python.exe feature_store/run_retrieval_demo.py
```

### D. Run Automated Test Suite
```powershell
.\.venv\Scripts\pytest.exe ml/tests/test_feast_feature_store.py -v
```

---

## 5. Schema Comparison: Feast vs. SentinelFeaturePipeline

| Dimension | `SentinelFeaturePipeline` (Active Model) | `Feast` (`account_hourly_stats`) |
| :--- | :--- | :--- |
| **Scope** | Single-transaction instantaneous signals | Historical multi-hour account behavior profile |
| **Total Features** | 20 features | 5 rolling features |
| **Features Included** | `amount`, `log_amount`, `oldbalanceOrg`, `log_oldbalanceOrg`, `oldbalanceDest`, `log_oldbalanceDest`, `amount_to_oldbalance_ratio`, `orig_balance_depletion`, `orig_oldbalance_is_zero`, `dest_oldbalance_is_zero`, `is_dest_merchant`, `expected_newbalance_orig`, `orig_deficit`, `hour_of_day`, `day_of_week`, 5 one-hot transaction types | `rolling_avg_amount_24h`, `rolling_txn_count_24h`, `rolling_max_amount_24h`, `zero_balance_frequency_24h`, `merchant_interaction_ratio_24h` |
| **Storage Layer** | Stateless (computed on incoming JSON payload) | Stateful (Parquet offline / SQLite online) |
| **Production Serving** | Live in FastAPI `backend/app/services/inference_service.py` | Isolated demonstration repository (Phase 3) |

> [!IMPORTANT]
> **Production Boundary Notice**:
> The existing production inference path (`backend/app/services/inference_service.py`) and trained XGBoost / GraphSAGE models currently consume the exact 20 features produced by `SentinelFeaturePipeline`.
> To prevent any regression or schema mismatch in deployed production services, the Feast feature store has been built in an **isolated, verified module** (`feature_store/`).
> Full end-to-end integration into production scoring would require retraining the model with the 5 additional Feast features appended (total 25 features) and updating the production inference pipeline.
