import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  api,
  ApiError,
  resolveApiBaseUrl,
  buildApiUrl,
  PRODUCTION_API_FALLBACK,
  DEVELOPMENT_API_FALLBACK,
} from '../api/client';

describe('Sentinel API Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe('API URL Resolution & Path Normalization', () => {
    it('uses explicit environment URL when provided and trims trailing slashes', () => {
      expect(resolveApiBaseUrl('https://custom-backend.com/', false)).toBe('https://custom-backend.com');
      expect(resolveApiBaseUrl('https://custom-backend.com///', true)).toBe('https://custom-backend.com');
    });

    it('falls back to live Render production URL in production when env URL is missing', () => {
      expect(resolveApiBaseUrl(undefined, true)).toBe(PRODUCTION_API_FALLBACK);
      expect(resolveApiBaseUrl('', true)).toBe(PRODUCTION_API_FALLBACK);
      expect(resolveApiBaseUrl('   ', true)).toBe(PRODUCTION_API_FALLBACK);
    });

    it('falls back to localhost:8000 in development when env URL is missing', () => {
      expect(resolveApiBaseUrl(undefined, false)).toBe(DEVELOPMENT_API_FALLBACK);
      expect(resolveApiBaseUrl('', false)).toBe(DEVELOPMENT_API_FALLBACK);
      expect(resolveApiBaseUrl('   ', false)).toBe(DEVELOPMENT_API_FALLBACK);
    });

    it('joins base URL and endpoint correctly without duplicated /api/v1 prefixes', () => {
      // Standard base without /api/v1 and endpoint with /api/v1
      expect(buildApiUrl('https://nexvia-sentinel.onrender.com', '/api/v1/health')).toBe(
        'https://nexvia-sentinel.onrender.com/api/v1/health'
      );

      // Trailing slash on base
      expect(buildApiUrl('https://nexvia-sentinel.onrender.com/', '/api/v1/health')).toBe(
        'https://nexvia-sentinel.onrender.com/api/v1/health'
      );

      // Base URL configured WITH /api/v1 and endpoint with /api/v1 (avoids duplicate /api/v1)
      expect(buildApiUrl('https://nexvia-sentinel.onrender.com/api/v1', '/api/v1/health')).toBe(
        'https://nexvia-sentinel.onrender.com/api/v1/health'
      );
      expect(buildApiUrl('https://nexvia-sentinel.onrender.com/api/v1/', '/api/v1/health')).toBe(
        'https://nexvia-sentinel.onrender.com/api/v1/health'
      );

      // Base URL without /api/v1 and endpoint without /api/v1 (ensures /api/v1 is inserted)
      expect(buildApiUrl('https://nexvia-sentinel.onrender.com', '/health')).toBe(
        'https://nexvia-sentinel.onrender.com/api/v1/health'
      );

      // Base URL with /api/v1 and endpoint without /api/v1
      expect(buildApiUrl('https://nexvia-sentinel.onrender.com/api/v1', '/health')).toBe(
        'https://nexvia-sentinel.onrender.com/api/v1/health'
      );
    });
  });

  describe('HTTP Communication & Error Handling', () => {
    it('fetches system health successfully and calls constructed URL', async () => {
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

      const fetchSpy = vi.spyOn(global, 'fetch').mockResolvedValueOnce({
        ok: true,
        json: async () => mockHealth,
      } as Response);

      const result = await api.getHealth();
      expect(result.status).toBe('HEALTHY');
      expect(result.artifacts_loaded).toBe(true);
      expect(result.total_indexed_nodes).toBe(5000);

      // Ensure fetch was invoked with a valid URL ending in /api/v1/health
      expect(fetchSpy).toHaveBeenCalled();
      const calledUrl = fetchSpy.mock.calls[0][0] as string;
      expect(calledUrl.endsWith('/api/v1/health')).toBe(true);
      expect(calledUrl).not.toContain('/api/v1/api/v1');
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
});
