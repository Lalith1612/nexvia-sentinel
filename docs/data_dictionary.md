# Sentinel — PaySim Data Dictionary & Prediction-Time Feature Availability Matrix

## 1. Primary Dataset Schema

| Column Name | Raw Type | Optimized Type | Description | Source / Generator | Target / Leakage Risk | Prediction-Time Availability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `step` | `int64` | `int16` | Chronological hour of simulation (1 to 743, spanning ~31 calendar days). | Simulator clock | Safe | **Available**: Current timestamp / hour of transaction submission. |
| `type` | `object` | `category` | Transaction financial action (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`). | User action | Safe | **Available**: Required in payment request payload. |
| `amount` | `float64` | `float32` | Requested transaction amount in local currency units. | Transaction amount | Safe | **Available**: Present in payment authorization request. |
| `nameOrig` | `object` | `string` | Customer account identifier initiating the transaction (e.g. `C1231006815`). | Sender account DB | Safe | **Available**: Authenticated session / account ID of initiator. |
| `oldbalanceOrg` | `float64` | `float32` | Initial balance of sender account prior to transaction processing. | Ledger query | Safe | **Available**: Instantaneously retrievable from core banking balance ledger. |
| `newbalanceOrig` | `float64` | `float32` | Balance of sender account after transaction processing. | Ledger update | **Post-settlement artifact** | **Conditionally Available**: At pre-authorization scoring, post-transaction settlement has not yet executed. However, expected new balance is deterministically `oldbalanceOrg - amount`. The raw `newbalanceOrig` can be used to compute pre-settlement anomaly features (e.g., `is_account_drain`). |
| `nameDest` | `object` | `string` | Recipient account identifier (e.g. `M1979787155` for merchant, `C553264065` for customer). | Recipient address | Safe | **Available**: Specified beneficiary in transaction instruction. |
| `oldbalanceDest` | `float64` | `float32` | Initial balance of recipient before transaction. | Ledger / Interbank | **Partial** | **Available for Internal accounts**: Available if beneficiary is on the same internal ledger. For merchants (`M*`), PaySim records `0.0`. |
| `newbalanceDest` | `float64` | `float32` | Recipient balance after transaction. | Ledger update | **Post-settlement artifact** | **Unavailable in Real-Time interbank**: Post-settlement destination balance is determined after clearing. For pre-authorization scoring, models must not rely on `newbalanceDest` as an independent truth. |
| `isFraud` | `int64` | `int8` | Ground truth fraud label: `1` if transaction was unauthorized fraudulent transfer/cashout, `0` otherwise. | Fraud analysis | **GROUND TRUTH TARGET** | **NEVER AVAILABLE AT PREDICTION TIME**. Strictly excluded from all model inputs. |
| `isFlaggedFraud` | `int64` | `int8` | Legacy rule-based system flag: `1` if an unauthorized attempt transferred > 200,000 in a single transaction. | Legacy heuristic rule | **Rule Leakage / Redundant** | **EXCLUDED FROM MODEL**: Triggers in only 16 rows out of 6,362,620 transactions (0.19% recall). Excluded to prevent the model from learning a trivial rule-based shortcut. |

---

## 2. Engineered Features & Scoring-Time Feasibility Matrix

The following domain features are computed for tabular (XGBoost) and graph (PyG) inference. Each feature is designed strictly with prediction-time inputs:

| Feature Name | Mathematical Definition | Prediction-Time Input Fields Required | Real-Time Feasibility |
| :--- | :--- | :--- | :--- |
| `error_balance_orig` | `newbalanceOrig + amount - oldbalanceOrg` *(or expected liquidation discrepancy)* | `oldbalanceOrg`, `amount` | **Instant (< 1ms)**: Detects whether transaction attempts to withdraw more than available funds. |
| `error_balance_dest` | `oldbalanceDest + amount - newbalanceDest` | `oldbalanceDest`, `amount` | **Instant (< 1ms)**: Detects zero-destination anomalies common in mule accounts. |
| `amount_to_oldbalance_ratio` | `amount / (oldbalanceOrg + 1.0)` | `amount`, `oldbalanceOrg` | **Instant (< 1ms)**: Ratios near 1.0 indicate complete account draining (hallmark of unauthorized takeover). |
| `is_orig_emptied` | `(oldbalanceOrg > 0) & (newbalanceOrig == 0)` *(or `amount >= oldbalanceOrg`)* | `oldbalanceOrg`, `amount` | **Instant (< 1ms)**: Binary indicator of balance depletion. |
| `is_dest_merchant` | `1 if nameDest.startswith('M') else 0` | `nameDest` | **Instant (< 1ms)**: Distinguishes P2M payments from P2P cash transfers. |
| `hour_of_day` | `step % 24` | `step` | **Instant (< 1ms)**: Cyclical diurnal pattern (nighttime vs business hours). |
| `day_of_week` | `(step // 24) % 7` | `step` | **Instant (< 1ms)**: Weekly cyclical seasonality. |
| `orig_txn_count_1h` | Rolling 1-hour transaction frequency for `nameOrig` | `nameOrig`, transaction cache | **Sub-millisecond via Feast**: Maintained in Redis/memory key-value store. |
| `dest_in_degree_24h` | Number of distinct senders sending to `nameDest` within 24 hours | `nameDest`, transaction cache | **Sub-millisecond via Feast / Graph**: Identifies high-fan-in mule accounts. |

---

## 3. Strict Chronological Split Definitions

To prevent future lookahead leakage, data is split sequentially on the simulation timeline (`step`):

```text
Step 1 ────────────────────────── Step 500 ─────── Step 620 ──────── Step 743
[          TRAINING WINDOW           ] [   VALIDATION   ] [     TEST HOLD-OUT   ]
           Days 1 to 21                     Days 22 to 26          Days 27 to 31
         6,061,807 txns                      209,984 txns            90,829 txns
          5,561 frauds                        1,286 frauds            1,366 frauds
        (0.0917% fraud)                     (0.6124% fraud)         (1.5039% fraud)
```

- **No Overlapping Steps**: $\text{Train}_{\max}(500) < \text{Val}_{\min}(501) \le \text{Val}_{\max}(620) < \text{Test}_{\min}(621)$.
- **Test Integrity**: The test partition represents the real-time holdout evaluation window and is **never sampled or artificially balanced**.
