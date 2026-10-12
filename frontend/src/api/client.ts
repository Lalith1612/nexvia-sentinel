/**
 * Sentinel API Client
 * Centralized HTTP communication layer with strong typing and error handling.
 */

import {
  AlertFilters,
  AlertListResponse,
  AlertResponse,
  AlertUpdateRequest,
  FraudPredictionResponse,
  HealthResponse,
  ModelInfoResponse,
  ReplayJobCreateRequest,
  ReplayJobListResponse,
  ReplayJobResponse,
  TransactionRequest,
} from '../types';

export class ApiError extends Error {
  public status: number;
  public details?: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...options.headers,
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorBody: any;
      try {
        errorBody = await response.json();
      } catch {
        errorBody = { message: await response.text() };
      }

      const message =
        errorBody?.detail ||
        errorBody?.message ||
        `Request failed with status ${response.status}`;

      throw new ApiError(
        typeof message === 'string' ? message : JSON.stringify(message),
        response.status,
        errorBody?.details || errorBody
      );
    }

    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    const networkMsg = (error as Error).message || 'Network connection to Sentinel backend failed';
    throw new ApiError(networkMsg, 0);
  }
}

export const api = {
  // System Health & Models
  getHealth: (): Promise<HealthResponse> => {
    return request<HealthResponse>('/api/v1/health');
  },

  getModelInfo: (): Promise<ModelInfoResponse> => {
    return request<ModelInfoResponse>('/api/v1/model/info');
  },

  // Fraud Inference
  predictFraud: (payload: TransactionRequest): Promise<FraudPredictionResponse> => {
    return request<FraudPredictionResponse>('/api/v1/predict/fraud', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  // Alerts Management
  getAlerts: (filters: AlertFilters = {}): Promise<AlertListResponse> => {
    const params = new URLSearchParams();
    if (filters.status) params.append('status', filters.status);
    if (filters.risk_tier) params.append('risk_tier', filters.risk_tier);
    if (filters.min_score !== undefined && filters.min_score !== null) {
      params.append('min_score', filters.min_score.toString());
    }
    if (filters.limit) params.append('limit', filters.limit.toString());
    if (filters.offset !== undefined) params.append('offset', filters.offset.toString());

    const qs = params.toString();
    return request<AlertListResponse>(`/api/v1/alerts${qs ? `?${qs}` : ''}`);
  },

  getAlert: (alertId: string): Promise<AlertResponse> => {
    return request<AlertResponse>(`/api/v1/alerts/${encodeURIComponent(alertId)}`);
  },

  updateAlert: (alertId: string, update: AlertUpdateRequest): Promise<AlertResponse> => {
    return request<AlertResponse>(`/api/v1/alerts/${encodeURIComponent(alertId)}`, {
      method: 'PATCH',
      body: JSON.stringify(update),
    });
  },

  // Replay Engine
  createReplayJob: (payload: ReplayJobCreateRequest): Promise<ReplayJobResponse> => {
    return request<ReplayJobResponse>('/api/v1/replay/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  getReplayJobs: (limit = 20, offset = 0): Promise<ReplayJobListResponse> => {
    return request<ReplayJobListResponse>(`/api/v1/replay/jobs?limit=${limit}&offset=${offset}`);
  },

  getReplayJob: (jobId: string): Promise<ReplayJobResponse> => {
    return request<ReplayJobResponse>(`/api/v1/replay/jobs/${encodeURIComponent(jobId)}`);
  },

  cancelReplayJob: (jobId: string): Promise<ReplayJobResponse> => {
    return request<ReplayJobResponse>(`/api/v1/replay/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    });
  },
};
