import React, { useEffect, useState } from 'react';
import {
  Server,
  Layers,
  ShieldAlert,
  Cpu,
  ArrowRight,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Activity,
  Sliders,
} from 'lucide-react';
import { api, getBaseUrl } from '../api/client';
import { HealthResponse, ModelInfoResponse, AlertListResponse, ReplayJobListResponse } from '../types';
import { HealthBadge, RiskTierBadge, ActionBadge, AlertStatusBadge, ReplayStatusBadge } from '../components/Badge';
import { LoadingState, ErrorAlert } from '../components/Feedback';
import { NavTab } from '../components/Sidebar';

interface OverviewPageProps {
  onNavigate: (tab: NavTab) => void;
  onRefreshTrigger?: number;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({ onNavigate, onRefreshTrigger }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [alertsData, setAlertsData] = useState<AlertListResponse | null>(null);
  const [recentJobs, setRecentJobs] = useState<ReplayJobListResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchOverviewData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [hRes, mRes, aRes, jRes] = await Promise.allSettled([
        api.getHealth(),
        api.getModelInfo(),
        api.getAlerts({ limit: 10 }),
        api.getReplayJobs(5, 0),
      ]);

      if (hRes.status === 'fulfilled') {
        setHealth(hRes.value);
      } else {
        throw new Error(hRes.reason?.message || 'Failed to connect to Sentinel Health API');
      }

      if (mRes.status === 'fulfilled') {
        setModelInfo(mRes.value);
      }

      if (aRes.status === 'fulfilled') {
        setAlertsData(aRes.value);
      }

      if (jRes.status === 'fulfilled') {
        setRecentJobs(jRes.value);
      }
    } catch (err: any) {
      setError(err.message || 'Unable to load Sentinel overview data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchOverviewData();
  }, [onRefreshTrigger]);

  if (isLoading && !health) {
    return <LoadingState message="Connecting to Nexvia Sentinel fraud core..." />;
  }

  if (error && !health) {
    return (
      <ErrorAlert
        title="Backend Connectivity Error"
        message={`Could not reach the Sentinel FastAPI backend at ${getBaseUrl()}. ${error}`}
        onRetry={fetchOverviewData}
      />
    );
  }

  // Authoritative dynamic ensemble weights derived from backend model-info
  const xgbWeight = modelInfo?.ensemble_weights?.tabular_xgboost !== undefined
    ? modelInfo.ensemble_weights.tabular_xgboost
    : null;
  const gnnWeight = modelInfo?.ensemble_weights?.relational_graphsage !== undefined
    ? modelInfo.ensemble_weights.relational_graphsage
    : null;

  const totalAlerts = alertsData?.total ?? 0;
  const openAlerts = alertsData?.items.filter((a) => a.status === 'OPEN').length ?? 0;
  const investigatingAlerts = alertsData?.items.filter((a) => a.status === 'INVESTIGATING').length ?? 0;
  const criticalCount = alertsData?.items.filter((a) => a.risk_tier === 'CRITICAL').length ?? 0;
  const highCount = alertsData?.items.filter((a) => a.risk_tier === 'HIGH').length ?? 0;
  const mediumCount = alertsData?.items.filter((a) => a.risk_tier === 'MEDIUM').length ?? 0;
  const lowCount = alertsData?.items.filter((a) => a.risk_tier === 'LOW').length ?? 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Degraded Alert Banner */}
      {health && health.status !== 'HEALTHY' && (
        <div
          style={{
            background: 'var(--badge-warning-bg)',
            border: '1px solid var(--badge-warning-border)',
            borderRadius: 'var(--radius-sm)',
            padding: '0.65rem 1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.65rem',
            color: 'var(--accent-amber)',
            fontSize: '0.8rem',
          }}
        >
          <AlertTriangle size={16} />
          <span>
            Sentinel service state reported as <strong>{health.status}</strong>. Some models or state tables may be uninitialized.
          </span>
        </div>
      )}

      {/* Row 1 — Compact Information-Rich KPI Grid */}
      <div className="grid grid-cols-4">
        {/* KPI 1: Service Status */}
        <div className="kpi-tile">
          <div className="kpi-header">
            <span className="kpi-label">Service Health</span>
            <Server size={14} color="var(--accent-cyan)" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">{health?.status || 'Unknown'}</span>
            {health && <HealthBadge status={health.status} />}
          </div>
          <div className="kpi-meta">
            RSS: {health?.memory_rss_mb ? `${health.memory_rss_mb.toFixed(1)} MB` : 'N/A'} • Device: {health?.torch_device || 'CPU'}
          </div>
        </div>

        {/* KPI 2: Relational Graph State */}
        <div className="kpi-tile">
          <div className="kpi-header">
            <span className="kpi-label">Relational Graph</span>
            <Layers size={14} color="var(--accent-blue)" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">
              {health?.total_indexed_nodes !== undefined ? health.total_indexed_nodes.toLocaleString('en-US') : 'N/A'}
            </span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>nodes</span>
          </div>
          <div className="kpi-meta">
            Settled Edges: {health?.settled_edges_count?.toLocaleString('en-US') ?? 'N/A'} • Cutoff: Step {health?.snapshot_cutoff_step ?? 'N/A'}
          </div>
        </div>

        {/* KPI 3: Active Ledger Alerts */}
        <div className="kpi-tile">
          <div className="kpi-header">
            <span className="kpi-label">Ledger Alerts</span>
            <ShieldAlert size={14} color="var(--accent-red)" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value">{totalAlerts}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>total</span>
          </div>
          <div className="kpi-meta">
            Open: <strong style={{ color: openAlerts > 0 ? 'var(--accent-red)' : 'var(--text-primary)' }}>{openAlerts}</strong> • Under Review: <strong>{investigatingAlerts}</strong>
          </div>
        </div>

        {/* KPI 4: Dynamic Model Artifact Readiness */}
        <div className="kpi-tile">
          <div className="kpi-header">
            <span className="kpi-label">Ensemble Readiness</span>
            <Cpu size={14} color="var(--accent-green)" />
          </div>
          <div className="kpi-value-row">
            <span className="kpi-value" style={{ fontSize: '1.05rem', marginTop: '2px' }}>
              {health?.artifacts_loaded ? 'Dual Artifacts Active' : 'Artifacts Missing'}
            </span>
          </div>
          <div className="kpi-meta">
            {xgbWeight !== null && gnnWeight !== null ? (
              <span>
                XGBoost ({(xgbWeight * 100).toFixed(0)}%) + GraphSAGE ({(gnnWeight * 100).toFixed(0)}%) Platt Calibrated
              </span>
            ) : (
              <span>Dynamic Dual Calibration Model</span>
            )}
          </div>
        </div>
      </div>

      {/* Row 2 — Analytical Panels (Balanced 2-Column Grid) */}
      <div className="grid grid-cols-2">
        {/* Model Architecture & Authoritative Parameters */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <Cpu size={15} color="var(--accent-cyan)" />
                <span>Production Model Architecture</span>
              </div>
              <div className="card-subtitle">Authoritative ensemble parameters returned by Sentinel Core</div>
            </div>
            <span className="badge badge-info">Phase 3 Validated</span>
          </div>

          {modelInfo ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {/* Architecture & Calibration Metadata */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.65rem' }}>
                <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Architecture</div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, marginTop: '2px' }}>{modelInfo.model_architecture}</div>
                </div>
                <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Calibration Method</div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, marginTop: '2px' }}>{modelInfo.calibration_method}</div>
                </div>
              </div>

              {/* Dynamic Authoritative Ensemble Weights */}
              <div>
                <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                  Authoritative Ensemble Weights (Backend Config)
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                  {Object.entries(modelInfo.ensemble_weights || {}).map(([key, weight]) => (
                    <div key={key}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', marginBottom: '3px' }}>
                        <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{key}</span>
                        <span style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>{(weight * 100).toFixed(1)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${weight * 100}%`,
                            background: key.includes('xgboost') ? 'var(--accent-blue)' : 'var(--accent-cyan)',
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Authoritative Risk Tier Thresholds */}
              <div>
                <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                  Operational Risk Tier Cutoffs
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.45rem' }}>
                  {Object.entries(modelInfo.risk_tier_thresholds || {}).map(([tier, cutoff]) => (
                    <div
                      key={tier}
                      style={{
                        background: 'var(--bg-tertiary)',
                        padding: '0.45rem',
                        borderRadius: 'var(--radius-sm)',
                        textAlign: 'center',
                        border: '1px solid var(--border-subtle)',
                      }}
                    >
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{tier}</div>
                      <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-cyan)', marginTop: '2px', fontFamily: 'var(--font-mono)' }}>
                        ≥ {typeof cutoff === 'number' ? cutoff.toFixed(2) : cutoff}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Restrained Prototype Attribution */}
              <div
                style={{
                  background: 'rgba(0, 217, 255, 0.03)',
                  border: '1px solid rgba(0, 217, 255, 0.15)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.55rem 0.75rem',
                  fontSize: '0.7rem',
                  color: 'var(--text-secondary)',
                  lineHeight: '1.4',
                }}
              >
                <strong style={{ color: 'var(--accent-cyan)' }}>Nexvia Technologies Attribution:</strong> {modelInfo.operational_prototype_boundary || 'Evaluated under strict validation partition.'}
              </div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', padding: '1rem' }}>Model configuration unavailable.</div>
          )}
        </div>

        {/* Risk Distribution & Ledger Alerts Posture */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <ShieldAlert size={15} color="var(--accent-red)" />
                <span>Alert Risk Posture & Distribution</span>
              </div>
              <div className="card-subtitle">Real-time breakdown of transactions flagged in database</div>
            </div>
            <button
              onClick={() => onNavigate('alerts')}
              className="btn btn-secondary"
              style={{ padding: '0.25rem 0.55rem', fontSize: '0.72rem' }}
            >
              <span>Investigation Queue</span>
              <ArrowRight size={12} />
            </button>
          </div>

          {alertsData && alertsData.items.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {/* Distribution Bar */}
              <div>
                <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                  Risk Tier Composition (Sample: {alertsData.items.length} records)
                </div>
                <div style={{ display: 'flex', height: '10px', borderRadius: 'var(--radius-xs)', overflow: 'hidden', gap: '2px' }}>
                  {criticalCount > 0 && (
                    <div style={{ width: `${(criticalCount / alertsData.items.length) * 100}%`, background: 'var(--accent-red)' }} title={`Critical: ${criticalCount}`} />
                  )}
                  {highCount > 0 && (
                    <div style={{ width: `${(highCount / alertsData.items.length) * 100}%`, background: 'var(--accent-amber)' }} title={`High: ${highCount}`} />
                  )}
                  {mediumCount > 0 && (
                    <div style={{ width: `${(mediumCount / alertsData.items.length) * 100}%`, background: 'var(--accent-blue)' }} title={`Medium: ${mediumCount}`} />
                  )}
                  {lowCount > 0 && (
                    <div style={{ width: `${(lowCount / alertsData.items.length) * 100}%`, background: 'var(--accent-green)' }} title={`Low: ${lowCount}`} />
                  )}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                  <span><strong style={{ color: 'var(--accent-red)' }}>{criticalCount}</strong> Critical</span>
                  <span><strong style={{ color: 'var(--accent-amber)' }}>{highCount}</strong> High</span>
                  <span><strong style={{ color: 'var(--accent-blue)' }}>{mediumCount}</strong> Medium</span>
                  <span><strong style={{ color: 'var(--accent-green)' }}>{lowCount}</strong> Low</span>
                </div>
              </div>

              {/* Top Recent Alerts Table */}
              <div className="table-container" style={{ border: '1px solid var(--border-subtle)' }}>
                <table>
                  <thead>
                    <tr>
                      <th>Alert ID</th>
                      <th>Type</th>
                      <th>Amount</th>
                      <th>Score</th>
                      <th>Tier</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alertsData.items.slice(0, 4).map((alert) => (
                      <tr key={alert.alert_id}>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', color: 'var(--accent-cyan)' }}>
                          {alert.alert_id.length > 14 ? `${alert.alert_id.substring(0, 14)}...` : alert.alert_id}
                        </td>
                        <td><span className="badge badge-neutral">{alert.type}</span></td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem', fontWeight: 600 }}>
                          ${alert.amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem', fontWeight: 700 }}>
                          {alert.hybrid_risk_score.toFixed(4)}
                        </td>
                        <td><RiskTierBadge tier={alert.risk_tier} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '2.5rem 1rem',
                color: 'var(--text-muted)',
                gap: '0.5rem',
                textAlign: 'center',
              }}
            >
              <CheckCircle2 size={26} color="var(--accent-green)" />
              <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                Zero Pending Alerts Recorded
              </div>
              <div style={{ fontSize: '0.74rem', maxWidth: '320px' }}>
                The persistent ledger currently contains no high-risk transactions. High-risk transactions detected via API or streaming will populate here.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Row 3 — Operational Detail: Bounded Replay Jobs */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">
              <Clock size={15} color="var(--accent-blue)" />
              <span>Recent Bounded Replay Simulation Jobs</span>
            </div>
            <div className="card-subtitle">Non-interfering dry-run execution history with zero ledger mutations</div>
          </div>
          <button
            onClick={() => onNavigate('replay')}
            className="btn btn-secondary"
            style={{ padding: '0.25rem 0.55rem', fontSize: '0.72rem' }}
          >
            <span>Launch Replay</span>
            <ArrowRight size={12} />
          </button>
        </div>

        {recentJobs && recentJobs.items.length > 0 ? (
          <div className="table-container" style={{ border: '1px solid var(--border-subtle)' }}>
            <table>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>Records Processed</th>
                  <th>Throughput</th>
                  <th>Elapsed</th>
                  <th>Created At</th>
                </tr>
              </thead>
              <tbody>
                {recentJobs.items.map((job) => (
                  <tr key={job.job_id}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.76rem', color: 'var(--accent-cyan)' }}>
                      {job.job_id}
                    </td>
                    <td><span className="badge badge-neutral">{job.mode}</span></td>
                    <td><ReplayStatusBadge status={job.status} /></td>
                    <td style={{ fontSize: '0.76rem', fontFamily: 'var(--font-mono)' }}>
                      {job.processed_records} / {job.total_records}
                    </td>
                    <td style={{ fontSize: '0.76rem', fontFamily: 'var(--font-mono)' }}>
                      {job.progress?.throughput_records_per_sec !== null && job.progress?.throughput_records_per_sec !== undefined
                        ? `${job.progress.throughput_records_per_sec.toFixed(1)} rec/s`
                        : 'N/A'}
                    </td>
                    <td style={{ fontSize: '0.76rem', fontFamily: 'var(--font-mono)' }}>
                      {job.progress?.elapsed_duration_sec !== undefined
                        ? `${job.progress.elapsed_duration_sec.toFixed(2)}s`
                        : job.metrics_summary?.latency_ms?.total_duration_sec !== undefined
                        ? `${job.metrics_summary.latency_ms.total_duration_sec.toFixed(2)}s`
                        : 'N/A'}
                    </td>
                    <td style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                      {new Date(job.created_at).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '2rem 1rem',
              color: 'var(--text-muted)',
              gap: '0.5rem',
            }}
          >
            <Clock size={24} />
            <div style={{ fontSize: '0.8rem' }}>No recent replay jobs recorded in ledger.</div>
            <button
              onClick={() => onNavigate('replay')}
              className="btn btn-primary"
              style={{ fontSize: '0.74rem', padding: '0.35rem 0.75rem', marginTop: '0.25rem' }}
            >
              Launch Dry-Run Replay
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
