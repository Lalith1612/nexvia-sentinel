import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { OverviewPage } from '../pages/OverviewPage';
import { api } from '../api/client';

vi.mock('../api/client', () => ({
  api: {
    getHealth: vi.fn(),
    getModelInfo: vi.fn(),
    getAlerts: vi.fn(),
    getReplayJobs: vi.fn(),
  },
}));

describe('OverviewPage Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders loading state initially', () => {
    (api.getHealth as any).mockReturnValue(new Promise(() => {}));
    (api.getModelInfo as any).mockReturnValue(new Promise(() => {}));
    (api.getAlerts as any).mockReturnValue(new Promise(() => {}));
    (api.getReplayJobs as any).mockReturnValue(new Promise(() => {}));

    render(<OverviewPage onNavigate={vi.fn()} />);
    expect(screen.getByText(/Connecting to.*Sentinel/i)).toBeInTheDocument();
  });

  it('renders health telemetry and model specs on success', async () => {
    (api.getHealth as any).mockResolvedValue({
      status: 'HEALTHY',
      service_name: 'Sentinel Anti-Fraud Engine',
      version: '1.0.0',
      artifacts_loaded: true,
      graph_state_initialized: true,
      snapshot_cutoff_step: 620,
      max_settled_step: 620,
      total_indexed_nodes: 10420,
      settled_edges_count: 24500,
      torch_device: 'cpu',
      memory_rss_mb: 512.4,
    });

    (api.getModelInfo as any).mockResolvedValue({
      model_architecture: 'Dual-Stage Tabular + Relational Ensemble',
      ensemble_weights: { tabular_xgboost: 0.8, relational_graphsage: 0.2 },
      risk_tier_thresholds: { LOW: 0.0, MEDIUM: 0.4, HIGH: 0.7, CRITICAL: 0.85 },
      tier_actions: { CRITICAL: 'AUTOMATED_BLOCK' },
      calibration_method: 'Platt Scaling (Isotonic Monotonicity)',
      validation_dataset: 'Phase 3 Clean Validation',
      governance_disclaimers: [],
      operational_prototype_boundary: 'Prototype operations console',
    });

    (api.getAlerts as any).mockResolvedValue({
      total: 0,
      limit: 10,
      offset: 0,
      items: [],
    });

    (api.getReplayJobs as any).mockResolvedValue({
      total: 0,
      limit: 5,
      offset: 0,
      items: [],
    });

    render(<OverviewPage onNavigate={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('10,420')).toBeInTheDocument();
      expect(screen.getByText('Dual Artifacts Active')).toBeInTheDocument();
      expect(screen.getByText(/Platt Scaling/i)).toBeInTheDocument();
      expect(screen.getByText(/No recent replay jobs recorded in ledger/i)).toBeInTheDocument();
    });
  });

  it('renders backend error banner when health check fails', async () => {
    (api.getHealth as any).mockRejectedValue(new Error('Connection refused'));
    (api.getModelInfo as any).mockRejectedValue(new Error('Connection refused'));
    (api.getAlerts as any).mockRejectedValue(new Error('Connection refused'));
    (api.getReplayJobs as any).mockRejectedValue(new Error('Connection refused'));

    render(<OverviewPage onNavigate={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/Backend Connectivity Error/i)).toBeInTheDocument();
      expect(screen.getByText(/Could not reach the Sentinel FastAPI backend/i)).toBeInTheDocument();
    });
  });
});
