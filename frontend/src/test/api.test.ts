import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api, ApiError } from '../api/client';

describe('Sentinel API Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('fetches system health successfully', async () => {
    const mockHealth = {
      status: 'HEALTHY',
      service_name: 'Sentinel Anti-Fraud Engine',
      version: '1.0.0',
      artifacts_loaded: true,
      graph_state_initialized: true,
      snapshot_cutoff_step: 620,
      max_settled_step: 620,
      total_indexed_nodes: 5000,
      settled_edges_count: 12000,
      torch_device: 'cpu',
      memory_rss_mb: 450.5,
    };

    vi.spyOn(global, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockHealth,
    } as Response);

    const result = await api.getHealth();
    expect(result.status).toBe('HEALTHY');
    expect(result.artifacts_loaded).toBe(true);
    expect(result.total_indexed_nodes).toBe(5000);
  });

  it('throws ApiError with server detail message on non-200 response', async () => {
    vi.spyOn(global, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'Temporal lookahead step must be >= 621' }),
    } as Response);

    await expect(
      api.predictFraud({
        step: 610,
        type: 'TRANSFER',
        amount: 100,
        nameOrig: 'C1',
        oldbalanceOrg: 100,
        nameDest: 'C2',
        oldbalanceDest: 0,
      })
    ).rejects.toThrow('Temporal lookahead step must be >= 621');
  });

  it('handles network failure with a clean ApiError', async () => {
    vi.spyOn(global, 'fetch').mockRejectedValueOnce(new Error('Failed to fetch'));

    await expect(api.getHealth()).rejects.toThrow(ApiError);
  });
});
