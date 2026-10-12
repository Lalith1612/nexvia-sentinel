import React, { useState } from 'react';
import {
  Send,
  Cpu,
  Layers,
  Sparkles,
  Info,
  Clock,
  ShieldAlert,
  Sliders,
  CheckCircle,
} from 'lucide-react';
import { api } from '../api/client';
import {
  FraudPredictionResponse,
  TransactionRequest,
  TransactionType,
} from '../types';
import { RiskTierBadge, ActionBadge } from '../components/Badge';
import { ErrorAlert } from '../components/Feedback';

export const PredictionPage: React.FC = () => {
  // Transaction Input Form State
  const [step, setStep] = useState<number>(621);
  const [type, setType] = useState<TransactionType>('TRANSFER');
  const [amount, setAmount] = useState<number>(250000);
  const [nameOrig, setNameOrig] = useState<string>('C1982736451');
  const [oldbalanceOrg, setOldbalanceOrg] = useState<number>(250000);
  const [newbalanceOrig, setNewbalanceOrig] = useState<number>(0);
  const [nameDest, setNameDest] = useState<string>('C8891023412');
  const [oldbalanceDest, setOldbalanceDest] = useState<number>(0);
  const [newbalanceDest, setNewbalanceDest] = useState<number>(0);

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<FraudPredictionResponse | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  // Quick preset loader
  const loadPreset = (preset: 'high_risk' | 'benign_payment' | 'cash_out') => {
    if (preset === 'high_risk') {
      setStep(622);
      setType('TRANSFER');
      setAmount(450000);
      setNameOrig('C1982736451');
      setOldbalanceOrg(450000);
      setNewbalanceOrig(0);
      setNameDest('C8891023412');
      setOldbalanceDest(0);
      setNewbalanceDest(0);
    } else if (preset === 'benign_payment') {
      setStep(625);
      setType('PAYMENT');
      setAmount(48.5);
      setNameOrig('C1029384756');
      setOldbalanceOrg(1500.0);
      setNewbalanceOrig(1451.5);
      setNameDest('M9876543210');
      setOldbalanceDest(0);
      setNewbalanceDest(0);
    } else if (preset === 'cash_out') {
      setStep(623);
      setType('CASH_OUT');
      setAmount(120000);
      setNameOrig('C8891023412');
      setOldbalanceOrg(120000);
      setNewbalanceOrig(0);
      setNameDest('C5544332211');
      setOldbalanceDest(5000);
      setNewbalanceDest(125000);
    }
  };

  const validateInput = (): string | null => {
    if (step <= 620) {
      return 'Step must be ≥ 621. Steps ≤ 620 are preserved historical ledger state and will be rejected by the backend to prevent lookahead bias.';
    }
    if (amount <= 0) {
      return 'Transaction amount must be strictly greater than $0.00.';
    }
    if (!nameOrig.trim() || !nameDest.trim()) {
      return 'Origin and destination account identifiers are required.';
    }
    if (oldbalanceOrg < 0 || oldbalanceDest < 0) {
      return 'Account balances cannot be negative.';
    }
    return null;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const validationErr = validateInput();
    if (validationErr) {
      setFormError(validationErr);
      return;
    }

    setFormError(null);
    setIsSubmitting(true);

    try {
      const payload: TransactionRequest = {
        step,
        type,
        amount,
        nameOrig: nameOrig.trim(),
        oldbalanceOrg,
        newbalanceOrig: newbalanceOrig !== null && !isNaN(newbalanceOrig) ? newbalanceOrig : undefined,
        nameDest: nameDest.trim(),
        oldbalanceDest,
        newbalanceDest: newbalanceDest !== null && !isNaN(newbalanceDest) ? newbalanceDest : undefined,
      };

      const result = await api.predictFraud(payload);
      setPrediction(result);
    } catch (err: any) {
      setFormError(err.message || 'Inference execution failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const xgbWeightPct = prediction
    ? (prediction.score_components.tabular_xgboost.ensemble_weight * 100).toFixed(0)
    : null;
  const gnnWeightPct = prediction
    ? (prediction.score_components.relational_graphsage.ensemble_weight * 100).toFixed(0)
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Top Banner explaining pre-authorization scope */}
      <div
        style={{
          background: 'rgba(0, 217, 255, 0.04)',
          border: '1px solid rgba(0, 217, 255, 0.2)',
          borderRadius: 'var(--radius-sm)',
          padding: '0.75rem 1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.65rem',
          fontSize: '0.78rem',
          color: 'var(--text-secondary)',
        }}
      >
        <Info size={16} color="var(--accent-cyan)" style={{ flexShrink: 0 }} />
        <div>
          This pre-authorization engine evaluates risk against the dual-stage calibrated ensemble. Predictions are purely advisory computations evaluated in-memory with strictly{' '}
          <strong style={{ color: 'var(--text-primary)' }}>zero ledger mutations</strong>.
        </div>
      </div>

      <div className="grid grid-cols-2">
        {/* Left: Input Form */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <Sliders size={15} color="var(--accent-cyan)" />
                <span>Transaction Parameters</span>
              </div>
              <div className="card-subtitle">Attributes evaluated by pre-authorization inference core</div>
            </div>
          </div>

          {/* Quick presets */}
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.35rem', fontWeight: 600 }}>
              Quick Scenario Presets:
            </div>
            <div style={{ display: 'flex', gap: '0.45rem', flexWrap: 'wrap' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => loadPreset('high_risk')}
                style={{ fontSize: '0.72rem', padding: '0.3rem 0.6rem' }}
              >
                <Sparkles size={12} color="var(--accent-red)" />
                <span>High-Risk Drain</span>
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => loadPreset('cash_out')}
                style={{ fontSize: '0.72rem', padding: '0.3rem 0.6rem' }}
              >
                <span>Suspicious Cash-Out</span>
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => loadPreset('benign_payment')}
                style={{ fontSize: '0.72rem', padding: '0.3rem 0.6rem' }}
              >
                <span>Benign Merchant Payment</span>
              </button>
            </div>
          </div>

          <form noValidate onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {formError && <ErrorAlert message={formError} />}

            <div className="grid grid-cols-2">
              <div className="form-group">
                <label className="form-label" htmlFor="tx-step">
                  Temporal Step (≥ 621) *
                </label>
                <input
                  id="tx-step"
                  type="number"
                  className="form-control"
                  value={step}
                  min={621}
                  onChange={(e) => setStep(parseInt(e.target.value) || 621)}
                  required
                />
                <span style={{ fontSize: '0.66rem', color: 'var(--text-muted)' }}>
                  Ledger snapshot cutoff is step 620
                </span>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="tx-type">Transaction Type *</label>
                <select
                  id="tx-type"
                  className="form-control"
                  value={type}
                  onChange={(e) => setType(e.target.value as TransactionType)}
                >
                  <option value="TRANSFER">TRANSFER</option>
                  <option value="CASH_OUT">CASH_OUT</option>
                  <option value="PAYMENT">PAYMENT</option>
                  <option value="CASH_IN">CASH_IN</option>
                  <option value="DEBIT">DEBIT</option>
                </select>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="tx-amount">Amount (USD) *</label>
              <input
                id="tx-amount"
                type="number"
                step="0.01"
                min="0.01"
                className="form-control"
                value={amount}
                onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
                required
              />
            </div>

            {/* Origin Account Details */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--accent-blue)', marginBottom: '0.45rem' }}>
                Origin Account (Sender)
              </div>
              <div className="grid grid-cols-3">
                <div className="form-group">
                  <label className="form-label" htmlFor="tx-orig-name">ID *</label>
                  <input
                    id="tx-orig-name"
                    type="text"
                    className="form-control"
                    value={nameOrig}
                    onChange={(e) => setNameOrig(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor="tx-orig-old">Old Balance</label>
                  <input
                    id="tx-orig-old"
                    type="number"
                    step="0.01"
                    className="form-control"
                    value={oldbalanceOrg}
                    onChange={(e) => setOldbalanceOrg(parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor="tx-orig-new">New Balance</label>
                  <input
                    id="tx-orig-new"
                    type="number"
                    step="0.01"
                    className="form-control"
                    value={newbalanceOrig}
                    onChange={(e) => setNewbalanceOrig(parseFloat(e.target.value) || 0)}
                  />
                </div>
              </div>
            </div>

            {/* Destination Account Details */}
            <div style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--accent-cyan)', marginBottom: '0.45rem' }}>
                Destination Account (Receiver)
              </div>
              <div className="grid grid-cols-3">
                <div className="form-group">
                  <label className="form-label" htmlFor="tx-dest-name">ID *</label>
                  <input
                    id="tx-dest-name"
                    type="text"
                    className="form-control"
                    value={nameDest}
                    onChange={(e) => setNameDest(e.target.value)}
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor="tx-dest-old">Old Balance</label>
                  <input
                    id="tx-dest-old"
                    type="number"
                    step="0.01"
                    className="form-control"
                    value={oldbalanceDest}
                    onChange={(e) => setOldbalanceDest(parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label" htmlFor="tx-dest-new">New Balance</label>
                  <input
                    id="tx-dest-new"
                    type="number"
                    step="0.01"
                    className="form-control"
                    value={newbalanceDest}
                    onChange={(e) => setNewbalanceDest(parseFloat(e.target.value) || 0)}
                  />
                </div>
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSubmitting}
              style={{ width: '100%', justifyContent: 'center', padding: '0.55rem', marginTop: '0.25rem' }}
            >
              <Send size={14} />
              <span>{isSubmitting ? 'Evaluating Ensemble...' : 'Run Fraud Inference'}</span>
            </button>
          </form>
        </div>

        {/* Right: Prediction Output */}
        <div>
          {prediction ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {/* Main Score Banner */}
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                      Ensemble Hybrid Risk Score
                    </div>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.65rem', marginTop: '0.2rem' }}>
                      <span
                        style={{
                          fontSize: '2.4rem',
                          fontWeight: 800,
                          fontFamily: 'var(--font-mono)',
                          color:
                            prediction.assigned_risk_tier === 'CRITICAL'
                              ? 'var(--accent-red)'
                              : prediction.assigned_risk_tier === 'HIGH'
                              ? 'var(--accent-amber)'
                              : prediction.assigned_risk_tier === 'MEDIUM'
                              ? 'var(--accent-blue)'
                              : 'var(--accent-green)',
                        }}
                      >
                        {prediction.hybrid_risk_score.toFixed(4)}
                      </span>
                      <RiskTierBadge tier={prediction.assigned_risk_tier} />
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                      Model Recommendation
                    </div>
                    <ActionBadge action={prediction.recommended_action} />
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '1rem', fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.45rem' }}>
                  <span>Tx ID: <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{prediction.transaction_id}</strong></span>
                  <span>Latency: <strong style={{ fontFamily: 'var(--font-mono)' }}>{prediction.latency_ms.toFixed(1)} ms</strong></span>
                </div>

                {/* Governance Alert Box */}
                <div
                  style={{
                    background: 'rgba(239, 68, 68, 0.08)',
                    border: '1px solid rgba(239, 68, 68, 0.25)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '0.65rem 0.85rem',
                    marginTop: '0.75rem',
                    fontSize: '0.72rem',
                    color: 'var(--text-secondary)',
                    lineHeight: 1.4,
                  }}
                >
                  <strong style={{ color: 'var(--accent-red)' }}>Model Advisory Disclaimer:</strong> This score is a statistical probability output. It does NOT constitute legal proof of fraud or confirmation that a transaction was blocked.
                </div>
              </div>

              {/* Sub-models Decomposition */}
              <div className="card">
                <div className="card-header">
                  <div className="card-title">
                    <Cpu size={14} color="var(--accent-cyan)" />
                    <span>Ensemble Decomposition</span>
                  </div>
                  <div className="card-subtitle">
                    {xgbWeightPct !== null && gnnWeightPct !== null
                      ? `${xgbWeightPct}% Tabular XGBoost + ${gnnWeightPct}% Relational GraphSAGE`
                      : 'Dual Calibrated Ensemble'}
                  </div>
                </div>

                <div className="grid grid-cols-2">
                  {/* XGBoost Tabular Component */}
                  <div style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent-blue)', marginBottom: '0.4rem', fontWeight: 600, fontSize: '0.78rem' }}>
                      <Cpu size={14} />
                      <span>Tabular XGBoost ({xgbWeightPct || '78'}%)</span>
                    </div>
                    <div style={{ fontSize: '0.72rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                      <div>Raw Score: <strong style={{ fontFamily: 'var(--font-mono)' }}>{prediction.score_components.tabular_xgboost.raw_score.toFixed(4)}</strong></div>
                      <div>Calibrated Prob: <strong style={{ fontFamily: 'var(--font-mono)' }}>{prediction.score_components.tabular_xgboost.calibrated_probability.toFixed(4)}</strong></div>
                      <div>Weighted Impact: <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)' }}>{(prediction.score_components.tabular_xgboost.weighted_contribution * 100).toFixed(2)}%</strong></div>
                    </div>
                  </div>

                  {/* GraphSAGE Relational Component */}
                  <div style={{ background: 'var(--bg-tertiary)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent-cyan)', marginBottom: '0.4rem', fontWeight: 600, fontSize: '0.78rem' }}>
                      <Layers size={14} />
                      <span>Relational GraphSAGE ({gnnWeightPct || '22'}%)</span>
                    </div>
                    <div style={{ fontSize: '0.72rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                      <div>Raw Score: <strong style={{ fontFamily: 'var(--font-mono)' }}>{prediction.score_components.relational_graphsage.raw_score.toFixed(4)}</strong></div>
                      <div>Calibrated Prob: <strong style={{ fontFamily: 'var(--font-mono)' }}>{prediction.score_components.relational_graphsage.calibrated_probability.toFixed(4)}</strong></div>
                      <div>Weighted Impact: <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{(prediction.score_components.relational_graphsage.weighted_contribution * 100).toFixed(2)}%</strong></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* SHAP Feature Attributions */}
              <div className="card">
                <div className="card-header">
                  <div className="card-title">
                    <Sparkles size={14} color="var(--accent-amber)" />
                    <span>SHAP Feature Explanations</span>
                  </div>
                  <div className="card-subtitle">Key drivers influencing the XGBoost tabular model</div>
                </div>

                {prediction.tabular_shap_attributions && prediction.tabular_shap_attributions.length > 0 ? (
                  <div className="table-container" style={{ border: '1px solid var(--border-subtle)' }}>
                    <table>
                      <thead>
                        <tr>
                          <th>Feature</th>
                          <th>Value</th>
                          <th>SHAP Impact</th>
                          <th>Risk Direction</th>
                        </tr>
                      </thead>
                      <tbody>
                        {prediction.tabular_shap_attributions.map((attr, idx) => (
                          <tr key={idx}>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem' }}>
                              {attr.feature}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem' }}>
                              {attr.value.toLocaleString('en-US')}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', fontWeight: 600 }}>
                              {attr.shap_contribution > 0 ? `+${attr.shap_contribution.toFixed(4)}` : attr.shap_contribution.toFixed(4)}
                            </td>
                            <td>
                              <span
                                className={`badge ${
                                  attr.impact === 'INCREASES_FRAUD_RISK'
                                    ? 'badge-danger'
                                    : 'badge-success'
                                }`}
                              >
                                {attr.impact === 'INCREASES_FRAUD_RISK' ? 'Increases Risk' : 'Reduces Risk'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.76rem', padding: '0.75rem' }}>
                    Feature attribution details unavailable.
                  </div>
                )}
              </div>

              {/* Topological Context */}
              {prediction.graph_topological_context && (
                <div className="card">
                  <div className="card-header">
                    <div className="card-title">
                      <Layers size={14} color="var(--accent-cyan)" />
                      <span>Graph Topological Context</span>
                    </div>
                    <div className="card-subtitle">Network degree, connectivity and inductive state</div>
                  </div>

                  <div className="grid grid-cols-2" style={{ fontSize: '0.72rem' }}>
                    <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                      <div>Origin Masked: <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{prediction.graph_topological_context.account_orig_masked}</strong></div>
                      <div>Out-Degree: <strong>{prediction.graph_topological_context.orig_historical_out_degree}</strong></div>
                      <div>Node Status: <span className="badge badge-neutral">{prediction.graph_topological_context.node_orig_status}</span></div>
                    </div>
                    <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                      <div>Dest Masked: <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>{prediction.graph_topological_context.account_dest_masked}</strong></div>
                      <div>In-Degree: <strong>{prediction.graph_topological_context.dest_historical_in_degree}</strong></div>
                      <div>Node Status: <span className="badge badge-neutral">{prediction.graph_topological_context.node_dest_status}</span></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div
              className="card"
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '4rem 2rem',
                textAlign: 'center',
                color: 'var(--text-muted)',
                gap: '0.75rem',
              }}
            >
              <Cpu size={36} color="var(--border-accent)" />
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                Awaiting Transaction Submission
              </div>
              <div style={{ fontSize: '0.78rem', maxWidth: '340px', lineHeight: 1.45 }}>
                Fill in the transaction parameters or select a scenario preset to run pre-authorization scoring against the dual-stage ensemble.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
