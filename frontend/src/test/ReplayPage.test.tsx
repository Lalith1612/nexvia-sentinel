import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ReplayPage } from '../pages/ReplayPage';
import { api } from '../api/client';
import { ReplayJobResponse } from '../types';

vi.mock('../api/client', () => ({
  api: {
    getReplayJobs: vi.fn(),
    getReplayJob: vi.fn(),
    createReplayJob: vi.fn(),
    cancelReplayJob: vi.fn(),
  },
}));

const mockCompletedJob: ReplayJobResponse = {
  job_id: 'REP_comp_112233',
  idempotency_key: 'key_1',
  mode: 'DRY_RUN',
  status: 'COMPLETED',
  total_records: 50,
  processed_records: 50,
  failed_records: 0,
  skipped_records: 0,
  progress: {
    progress_percentage: 100.0,
    elapsed_duration_sec: 1.25,
    throughput_records_per_sec: 40.0,
    estimated_remaining_sec: 0.0,
  },
  metrics_summary: {
    records_summary: {
      total: 50,
      processed: 50,
      failed: 0,
      skipped_stale_watermark: 0,
    },
    latency_ms: {
      min: 5.1,
      max: 22.3,
      mean: 11.2,
      p50: 10.4,
      p95: 18.9,
      total_duration_sec: 1.25,
    },
    financial_exposure: {
      total_amount_replayed: 1250000,
      flagged_amount: 350000,
      flagged_amount_percentage: 28.0,
      flagged_record_percentage: 12.0,
    },
    score_distribution: {
      mean_score: 0.22,
      median_score: 0.08,
      score_buckets: { '0.0-0.2': 40, '0.8-1.0': 6 },
    },
    risk_tier_counts: { LOW: 40, MEDIUM: 4, HIGH: 2, CRITICAL: 4 },
    action_counts: { AUTOMATED_CLEAR_AND_SETTLE: 40, AUTOMATED_BLOCK: 4 },
    state_mutations: {
      alerts_created: 0,
      settlements_applied: 0,
    },
    disclaimer: 'Dry run only',
  },
  created_at: '2026-10-12T00:00:00Z',
  updated_at: '2026-10-12T00:00:01Z',
};

const mockRunningJob: ReplayJobResponse = {
  job_id: 'REP_run_998877',
  idempotency_key: 'key_2',
  mode: 'DRY_RUN',
  status: 'RUNNING',
  total_records: 100,
  processed_records: 45,
  failed_records: 0,
  skipped_records: 0,
  progress: {
    progress_percentage: 45.0,
    elapsed_duration_sec: 0.9,
    throughput_records_per_sec: 50.0,
    estimated_remaining_sec: 1.1,
  },
  created_at: '2026-10-12T00:01:00Z',
  updated_at: '2026-10-12T00:01:01Z',
};

describe('ReplayPage Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders completed job details and non-interference verification', async () => {
    (api.getReplayJobs as any).mockResolvedValue({
      total: 1,
      limit: 20,
      offset: 0,
      items: [mockCompletedJob],
    });

    render(<ReplayPage />);

    await waitFor(() => {
      expect(screen.getAllByText('REP_comp_112233').length).toBeGreaterThan(0);
      expect(screen.getAllByText('COMPLETED').length).toBeGreaterThan(0);
      expect(screen.getByText('$1,250,000.00')).toBeInTheDocument();
      expect(screen.getByText(/Non-Interference Verification:/i)).toBeInTheDocument();
      expect(screen.getByText(/Job in terminal state \(COMPLETED\)/i)).toBeInTheDocument();
    });
  });

  it('allows cancellation for RUNNING jobs', async () => {
    (api.getReplayJobs as any).mockResolvedValue({
      total: 1,
      limit: 20,
      offset: 0,
      items: [mockRunningJob],
    });

    (api.cancelReplayJob as any).mockResolvedValue({
      ...mockRunningJob,
      status: 'CANCELLED',
    });

    render(<ReplayPage />);

    await waitFor(() => {
      expect(screen.getByText('Cancel Replay Job')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Cancel Replay Job'));

    await waitFor(() => {
      expect(screen.getByText('Confirm Cancel')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Confirm Cancel'));

    await waitFor(() => {
      expect(api.cancelReplayJob).toHaveBeenCalledWith('REP_run_998877');
    });
  });
});
