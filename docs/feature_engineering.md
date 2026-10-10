# Sentinel — Feature Engineering Specification & Availability Matrix

## 1. Prediction Point & Real-Time Context
In the Sentinel payment architecture, fraud evaluation occurs at the **Pre-Authorization Scoring Point**:
- **Arrival Moment:** A customer submits a payment / transfer request via mobile app or web portal.
- **Available State:**
  - Transaction intent: `step` (timestamp), `type`, `amount`, `nameOrig` (authenticated sender), `nameDest` (beneficiary address).
  - Pre-transaction core ledger state: `oldbalanceOrg` (current account balance).
  - Beneficiary known state: `oldbalanceDest` (if internal account; 0.0 for external/merchant).
- **Unavailable State:**
  - Settlement ledger state: `newbalanceOrig`, `newbalanceDest`. These are only written *after* transaction authorization, execution, and clearing.
  - Fraud status: `isFraud` (ground truth label, assigned days/weeks later post-chargeback/investigation).
  - Heuristic flag: `isFlaggedFraud` (legacy rule flag, excluded to avoid rule leakage).

---

## 2. Feature Availability & Transformation Matrix

| Feature Identifier | Category | Formula / Definition | Required Raw Inputs | Prediction-Time Latency |
| :--- | :--- | :--- | :--- | :--- |
| `amount` | Core Financial | Raw transaction amount in local currency. | `amount` | Instant (< 1 µs) |
| `log_amount` | Transformed | $\log(1 + \text{amount})$ to compress heavy-tailed skewness. | `amount` | Instant (< 1 µs) |
| `oldbalanceOrg` | Ledger State | Available balance of originating account before transaction. | `oldbalanceOrg` | Instant (< 1 µs) |
| `log_oldbalanceOrg` | Transformed | $\log(1 + \text{oldbalanceOrg})$ | `oldbalanceOrg` | Instant (< 1 µs) |
| `oldbalanceDest` | Ledger State | Beneficiary balance prior to transfer (0.0 for merchants). | `oldbalanceDest` | Instant (< 1 µs) |
| `log_oldbalanceDest` | Transformed | $\log(1 + \text{oldbalanceDest})$ | `oldbalanceDest` | Instant (< 1 µs) |
| `amount_to_oldbalance_ratio` | Risk Ratio | $\frac{\text{amount}}{\text{oldbalanceOrg} + 1.0}$. Values $\approx 1.0$ indicate total account draining. | `amount`, `oldbalanceOrg` | Instant (< 1 µs) |
| `orig_balance_depletion` | Binary Indicator | $1.0 \iff (\text{amount} \ge \text{oldbalanceOrg} \land \text{oldbalanceOrg} > 0)$ | `amount`, `oldbalanceOrg` | Instant (< 1 µs) |
| `orig_oldbalance_is_zero` | Binary Indicator | $1.0 \iff \text{oldbalanceOrg} \le 0$ | `oldbalanceOrg` | Instant (< 1 µs) |
| `dest_oldbalance_is_zero` | Binary Indicator | $1.0 \iff \text{oldbalanceDest} \le 0$ | `oldbalanceDest` | Instant (< 1 µs) |
| `is_dest_merchant` | Entity Type | $1.0 \iff \text{nameDest starts with 'M'}$ | `nameDest` | Instant (< 1 µs) |
| `expected_newbalance_orig` | Projected Ledger | $\max(0.0, \text{oldbalanceOrg} - \text{amount})$ | `oldbalanceOrg`, `amount` | Instant (< 1 µs) |
| `orig_deficit` | Overdraft Risk | $\max(0.0, \text{amount} - \text{oldbalanceOrg})$ | `amount`, `oldbalanceOrg` | Instant (< 1 µs) |
| `hour_of_day` | Temporal Diurnal | $\text{step} \pmod{24}$ (Hour 0 to 23) | `step` | Instant (< 1 µs) |
| `day_of_week` | Temporal Weekly | $\lfloor\text{step} / 24\rfloor \pmod 7$ (Day 0 to 6) | `step` | Instant (< 1 µs) |
| `type_CASH_IN` | One-Hot Type | Indicator for Cash Deposit | `type` | Instant (< 1 µs) |
| `type_CASH_OUT` | One-Hot Type | Indicator for Cash Withdrawal (High-Risk Fraud Vector) | `type` | Instant (< 1 µs) |
| `type_DEBIT` | One-Hot Type | Indicator for Debit Operation | `type` | Instant (< 1 µs) |
| `type_PAYMENT` | One-Hot Type | Indicator for Merchant Payment | `type` | Instant (< 1 µs) |
| `type_TRANSFER` | One-Hot Type | Indicator for Account Transfer (High-Risk Fraud Vector) | `type` | Instant (< 1 µs) |

---

## 3. Strict Leakage Prevention Guarantees

1. **Target Isolation:** `isFraud` is strictly partitioned as the label array `y` and never admitted to feature matrix `X`.
2. **Rule Isolation:** `isFlaggedFraud` is strictly excluded. It represents a 16-row heuristic rule trigger that would introduce artificial shortcutting.
3. **Settlement Isolation:** `newbalanceOrig` and `newbalanceDest` are excluded from the pre-authorization feature set. Instead, the model relies on mathematically projected pre-authorization values (`expected_newbalance_orig`, `orig_balance_depletion`).
4. **Chronological Preservation:** Features do not incorporate future sliding windows or retrospective aggregations from later steps.
