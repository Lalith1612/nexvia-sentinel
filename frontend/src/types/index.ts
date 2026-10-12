/**
 * Sentinel Frontend TypeScript Definitions
 * Strictly aligned with backend FastAPI Pydantic schemas.
 */

// --- System & Model Information ---

export type ServiceHealthStatus = 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY';

export interface HealthResponse {
  status: ServiceHealthStatus;
  service_name: string;
  version: string;
  artifacts_loaded: boolean;
  graph_state_initialized: boolean;
  snapshot_cutoff_step: number;
  max_settled_step: number;
  total_indexed_nodes: number;
  settled_edges_count: number;
  torch_device: string;
  memory_rss_mb: number;
}

export interface ModelInfoResponse {
  model_architecture: string;
  ensemble_weights: Record<string, number>;
  risk_tier_thresholds: Record<string, number>;
  tier_actions: Record<string, string>;
  calibration_method: string;
  validation_dataset: string;
  governance_disclaimers: string[];
  operational_prototype_boundary: string;
}

// --- Fraud Inference ---

export type TransactionType = 'CASH_IN' | 'CASH_OUT' | 'DEBIT' | 'PAYMENT' | 'TRANSFER';
export type RiskTier = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
export type RecommendedAction =
  | 'AUTOMATED_BLOCK'
  | 'STEP_UP_AUTHENTICATION'
  | 'MANUAL_ANALYST_REVIEW'
  | 'AUTOMATED_CLEAR_AND_SETTLE';

export interface TransactionRequest {
  step: number;
  type: TransactionType;
  amount: number;
  nameOrig: string;
  oldbalanceOrg: number;
  newbalanceOrig?: number | null;
  nameDest: string;
  oldbalanceDest: number;
  newbalanceDest?: number | null;
}

export interface ScoreComponent {
  raw_score: number;
  calibrated_probability: number;
  ensemble_weight: number;
  weighted_contribution: number;
}

export interface FeatureAttribution {
  feature: string;
  value: number;
  shap_contribution: number;
  impact: 'INCREASES_FRAUD_RISK' | 'REDUCES_FRAUD_RISK';
}

export interface GraphTopologicalContext {
  account_orig_masked: string;
  account_dest_masked: string;
  is_destination_merchant: boolean;
  orig_historical_out_degree: number;
  orig_historical_out_volume: number;
  dest_historical_in_degree: number;
  dest_historical_in_volume: number;
  raw_gnn_anomaly_score: number;
  calibrated_gnn_risk: number;
  relational_weight_in_ensemble: number;
  node_orig_status: 'SETTLED_SNAPSHOT' | 'COLD_START_INDUCTIVE' | 'SETTLED_PENDING_REFRESH';
  node_dest_status: 'SETTLED_SNAPSHOT' | 'COLD_START_INDUCTIVE' | 'SETTLED_PENDING_REFRESH';
}

export interface FraudPredictionResponse {
  transaction_id: string;
  hybrid_risk_score: number;
  assigned_risk_tier: RiskTier;
  recommended_action: RecommendedAction;
  score_components: {
    tabular_xgboost: ScoreComponent;
    relational_graphsage: ScoreComponent;
    [key: string]: ScoreComponent;
  };
  tabular_shap_attributions: FeatureAttribution[];
  graph_topological_context: GraphTopologicalContext;
  regulatory_disclaimers: string[];
  latency_ms: number;
}

// --- Alerts Management ---

export type AlertStatus = 'OPEN' | 'INVESTIGATING' | 'CONFIRMED_FRAUD' | 'FALSE_POSITIVE' | 'CLOSED';

export interface AlertResponse {
  alert_id: string;
  transaction_id: string;
  step: number;
  type: string;
  amount: number;
  name_orig: string;
  name_dest: string;
  hybrid_risk_score: number;
  risk_tier: RiskTier;
  recommended_action: RecommendedAction;
  status: AlertStatus;
  analyst_id: string | null;
  resolution_notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface AlertUpdateRequest {
  status: AlertStatus;
  analyst_id?: string | null;
  resolution_notes?: string | null;
}

export interface AlertListResponse {
  total: number;
  limit: number;
  offset: number;
  items: AlertResponse[];
}

export interface AlertFilters {
  status?: AlertStatus;
  risk_tier?: RiskTier;
  min_score?: number;
  limit?: number;
  offset?: number;
}

// --- Transaction Replay Engine ---

export type ReplayJobStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';

export interface ReplayProgressStatus {
  progress_percentage: number;
  elapsed_duration_sec: number;
  throughput_records_per_sec: number | null;
  estimated_remaining_sec: number | null;
}

export interface LatencyMetrics {
  min: number;
  max: number;
  mean: number;
  p50: number;
  p95: number;
  total_duration_sec: number;
}

export interface FinancialExposureMetrics {
  total_amount_replayed: number;
  flagged_amount: number;
  flagged_amount_percentage: number;
  flagged_record_percentage: number;
}

export interface ScoreDistributionMetrics {
  mean_score: number | null;
  median_score: number | null;
  mean_hybrid_score?: number | null;
  median_hybrid_score?: number | null;
  score_buckets: Record<string, number>;
}

export interface StateMutationsSummary {
  alerts_created: number;
  settlements_applied: number;
}

export interface ReplayRecordsSummary {
  total: number;
  processed: number;
  failed: number;
  skipped_stale_watermark: number;
}

export interface ReplayMetricsSummary {
  records_summary: ReplayRecordsSummary;
  latency_ms: LatencyMetrics;
  progress?: ReplayProgressStatus | null;
  financial_exposure: FinancialExposureMetrics;
  score_distribution: ScoreDistributionMetrics;
  risk_tier_counts: Record<string, number>;
  action_counts: Record<string, number>;
  state_mutations: StateMutationsSummary;
  disclaimer: string;
}

export interface ReplayJobResponse {
  job_id: string;
  idempotency_key: string | null;
  mode: 'DRY_RUN';
  status: ReplayJobStatus;
  total_records: number;
  processed_records: number;
  failed_records: number;
  skipped_records: number;
  progress?: ReplayProgressStatus | null;
  metrics_summary?: ReplayMetricsSummary | null;
  error_summary?: Array<Record<string, unknown>> | null;
  created_at: string;
  updated_at: string;
}

export interface ReplayJobListResponse {
  total: number;
  limit: number;
  offset: number;
  items: ReplayJobResponse[];
}

export interface ReplayJobCreateRequest {
  mode?: 'DRY_RUN';
  idempotency_key?: string;
  dataset_source?: 'test.parquet';
  max_records?: number;
  records?: TransactionRequest[];
}
