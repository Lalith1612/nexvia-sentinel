import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { PredictionPage } from '../pages/PredictionPage';
import { api } from '../api/client';
import { FraudPredictionResponse } from '../types';

vi.mock('../api/client', () => ({
  api: {
    predictFraud: vi.fn(),
  },
}));

const mockPredictionResponse: FraudPredictionResponse = {
  transaction_id: 'TX_PRED_889900',
  hybrid_risk_score: 0.9632,
  assigned_risk_tier: 'CRITICAL',
  recommended_action: 'AUTOMATED_BLOCK',
  score_components: {
    tabular_xgboost: {
      raw_score: 0.95,
      calibrated_probability: 0.96,
      ensemble_weight: 0.8,
      weighted_contribution: 0.768,
    },
    relational_graphsage: {
      raw_score: 0.88,
      calibrated_probability: 0.90,
      ensemble_weight: 0.2,
      weighted_contribution: 0.18,
    },
  },
  tabular_shap_attributions: [
    {
      feature: 'amount',
      value: 450000,
      shap_contribution: 0.42,
      impact: 'INCREASES_FRAUD_RISK',
    },
  ],
  graph_topological_context: {
    account_orig_masked: 'C198***',
    account_dest_masked: 'C889***',
    is_destination_merchant: false,
    orig_historical_out_degree: 1,
    orig_historical_out_volume: 450000,
    dest_historical_in_degree: 0,
    dest_historical_in_volume: 0,
    raw_gnn_anomaly_score: 0.88,
    calibrated_gnn_risk: 0.90,
    relational_weight_in_ensemble: 0.2,
    node_orig_status: 'SETTLED_SNAPSHOT',
    node_dest_status: 'COLD_START_INDUCTIVE',
  },
  regulatory_disclaimers: ['Advisory only.'],
  latency_ms: 12.4,
};

describe('PredictionPage Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('validates step >= 621 before submitting', async () => {
    render(<PredictionPage />);

    const stepInput = screen.getByLabelText(/Temporal Step/i);
    fireEvent.change(stepInput, { target: { value: '615' } });

    const form = screen.getByRole('button', { name: /Run Fraud Inference/i }).closest('form')!;
    fireEvent.submit(form);

    expect(
      screen.getByText(/Step must be ≥ 621. Steps ≤ 620 are preserved historical ledger state/i)
    ).toBeInTheDocument();
    expect(api.predictFraud).not.toHaveBeenCalled();
  });

  it('submits valid prediction and displays score, disclaimers, and SHAP', async () => {
    (api.predictFraud as any).mockResolvedValue(mockPredictionResponse);

    render(<PredictionPage />);

    // Load High-Risk Drain preset
    fireEvent.click(screen.getByText('High-Risk Drain'));

    // Submit
    fireEvent.click(screen.getByRole('button', { name: /Run Fraud Inference/i }));

    await waitFor(() => {
      expect(screen.getByText('0.9632')).toBeInTheDocument();
      expect(screen.getByText('AUTOMATED BLOCK')).toBeInTheDocument();
      expect(screen.getByText(/Model Advisory Disclaimer:/i)).toBeInTheDocument();
      expect(screen.getByText('Tabular XGBoost (80%)')).toBeInTheDocument();
      expect(screen.getByText('Relational GraphSAGE (20%)')).toBeInTheDocument();
      expect(screen.getByText('12.4 ms')).toBeInTheDocument();
    });
  });
});
