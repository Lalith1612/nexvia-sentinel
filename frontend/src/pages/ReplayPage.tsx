import React, { useEffect, useState, useRef } from 'react';
import {
  History,
  Play,
  XCircle,
  RefreshCw,
  Clock,
  DollarSign,
  BarChart2,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  Activity,
  Layers,
  ChevronRight,
} from 'lucide-react';
import { api } from '../api/client';
import {
  ReplayJobCreateRequest,
  ReplayJobListResponse,
  ReplayJobResponse,
  ReplayJobStatus,
} from '../types';
import { ReplayStatusBadge } from '../components/Badge';
import { LoadingState, ErrorAlert } from '../components/Feedback';

interface ReplayPageProps {
  onRefreshTrigger?: number;
}

export const ReplayPage: React.FC<ReplayPageProps> = ({ onRefreshTrigger }) => {
  const [jobs, setJobs] = useState<ReplayJobResponse[]>([]);
  const [selectedJob, setSelectedJob] = useState<ReplayJobResponse | null>(null);
  const [totalJobs, setTotalJobs] = useState<number>(0);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Job Creation State
  const [isCreating, setIsCreating] = useState<boolean>(false);
  const [createMaxRecords, setCreateMaxRecords] = useState<number>(50);
  const [createDataset] = useState<'test.parquet'>('test.parquet');
  const [customKey, setCustomKey] = useState<string>('');
  const [createError, setCreateError] = useState<string | null>(null);

  // Cancellation State
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const [confirmingCancel, setConfirmingCancel] = useState<boolean>(false);
  const [cancelMessage, setCancelMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Polling ref for active job
  const pollTimerRef = useRef<number | null>(null);

  const fetchJobs = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res: ReplayJobListResponse = await api.getReplayJobs(20, 0);
      setJobs(res.items);
      setTotalJobs(res.total);
      setSelectedJob((prev) => {
        if (!prev) return res.items.length > 0 ? res.items[0] : null;
        const found = res.items.find((j) => j.job_id === prev.job_id);
        return found || prev;
      });
    } catch (err: any) {
      setError(err.message || 'Failed to retrieve replay jobs from Sentinel');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, [onRefreshTrigger]);

  // Polling for selected job if it is PENDING or RUNNING
  useEffect(() => {
    if (!selectedJob) return;

    const isRunning = selectedJob.status === 'RUNNING' || selectedJob.status === 'PENDING';
    if (!isRunning) {
      if (pollTimerRef.current) {
        window.clearTimeout(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }

    const poll = async () => {
      try {
        const updated = await api.getReplayJob(selectedJob.job_id);
        setSelectedJob(updated);
        setJobs((prev) => prev.map((j) => (j.job_id === updated.job_id ? updated : j)));

        if (updated.status === 'RUNNING' || updated.status === 'PENDING') {
          pollTimerRef.current = window.setTimeout(poll, 1500);
        }
      } catch (err) {
        console.error('Polling replay job failed', err);
      }
    };

    pollTimerRef.current = window.setTimeout(poll, 1500);

    return () => {
      if (pollTimerRef.current) {
        window.clearTimeout(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [selectedJob?.job_id, selectedJob?.status]);

  const handleCreateJob = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsCreating(true);
    setCreateError(null);

    try {
      const payload: ReplayJobCreateRequest = {
        mode: 'DRY_RUN',
        dataset_source: createDataset,
        max_records: createMaxRecords,
        idempotency_key: customKey.trim() ? customKey.trim() : undefined,
      };

      const newJob = await api.createReplayJob(payload);
      setJobs((prev) => [newJob, ...prev]);
      setSelectedJob(newJob);
      setCustomKey('');
    } catch (err: any) {
      setCreateError(err.message || 'Failed to submit replay job');
    } finally {
      setIsCreating(false);
    }
  };

  const handleCancelJob = async (jobId: string) => {
    setIsCancelling(true);
    setCancelMessage(null);

    try {
      const updated = await api.cancelReplayJob(jobId);
      setSelectedJob(updated);
      setJobs((prev) => prev.map((j) => (j.job_id === updated.job_id ? updated : j)));
      setCancelMessage({ type: 'success', text: `Job ${jobId} cancellation requested successfully.` });
    } catch (err: any) {
      setCancelMessage({
        type: 'error',
        text: err.status === 409
          ? 'Cannot cancel: Job is already in a terminal state.'
          : err.message || 'Cancellation request failed.',
      });
    } finally {
      setIsCancelling(false);
    }
  };

  const isSelectedCancellable =
    selectedJob && (selectedJob.status === 'RUNNING' || selectedJob.status === 'PENDING');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Top Advisory Banner */}
      <div
        style={{
          background: 'rgba(59, 130, 246, 0.05)',
          border: '1px solid rgba(59, 130, 246, 0.25)',
          borderRadius: 'var(--radius-md)',
          padding: '0.85rem 1.25rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          fontSize: '0.82rem',
          color: 'var(--text-secondary)',
        }}
      >
        <ShieldCheck size={20} color="var(--accent-blue)" style={{ flexShrink: 0 }} />
        <div>
          <strong style={{ color: 'var(--text-primary)' }}>Dry-Run Replay Isolation:</strong> Replay jobs evaluate historical transactions against the model in memory. They simulate pre-authorization decisions with strictly{' '}
          <strong style={{ color: 'var(--accent-green)' }}>zero ledger mutations</strong> and{' '}
          <strong style={{ color: 'var(--accent-green)' }}>zero alert writes</strong>.
        </div>
      </div>

      <div className="grid grid-cols-3">
        {/* Left Column: Create Job & Jobs List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Create Job Card */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Launch Simulation</div>
              <div className="card-subtitle">Batch dry-run over test dataset</div>
            </div>

            <form onSubmit={handleCreateJob} style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {createError && <ErrorAlert message={createError} />}

              <div className="form-group">
                <label className="form-label" htmlFor="replay-records-count">Max Records to Evaluate</label>
                <select
                  id="replay-records-count"
                  className="form-control"
                  value={createMaxRecords}
                  onChange={(e) => setCreateMaxRecords(parseInt(e.target.value))}
                >
                  <option value={10}>10 Records (Fast check)</option>
                  <option value={50}>50 Records (Standard validation)</option>
                  <option value={100}>100 Records</option>
                  <option value={250}>250 Records</option>
                  <option value={500}>500 Records</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="replay-idempotency-key">Custom Idempotency Key (Optional)</label>
                <input
                  id="replay-idempotency-key"
                  type="text"
                  className="form-control"
                  placeholder="e.g. rep_audit_test_01"
                  value={customKey}
                  onChange={(e) => setCustomKey(e.target.value)}
                />
              </div>

              <button
                type="submit"
                className="btn btn-primary"
                disabled={isCreating}
                style={{ width: '100%', justifyContent: 'center' }}
              >
                <Play size={14} />
                <span>{isCreating ? 'Submitting...' : 'Queue Dry-Run Job'}</span>
              </button>
            </form>
          </div>

          {/* Jobs List */}
          <div className="card" style={{ padding: 0 }}>
            <div style={{ padding: '0.85rem 1rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Recent Replay Jobs</span>
              <button
                className="btn btn-secondary"
                onClick={fetchJobs}
                disabled={isLoading}
                style={{ padding: '0.25rem 0.5rem' }}
                aria-label="Refresh job list"
              >
                <RefreshCw size={13} style={{ animation: isLoading ? 'spin 1s linear infinite' : 'none' }} />
              </button>
            </div>

            {isLoading && jobs.length === 0 ? (
              <LoadingState message="Fetching replay jobs..." />
            ) : jobs.length > 0 ? (
              <div style={{ maxHeight: '420px', overflowY: 'auto' }}>
                {jobs.map((job) => {
                  const isSelected = selectedJob?.job_id === job.job_id;
                  return (
                    <div
                      key={job.job_id}
                      onClick={() => setSelectedJob(job)}
                      style={{
                        padding: '0.75rem 1rem',
                        borderBottom: '1px solid var(--border-color)',
                        cursor: 'pointer',
                        background: isSelected ? 'var(--bg-tertiary)' : 'transparent',
                        borderLeft: isSelected ? '3px solid var(--accent-blue)' : '3px solid transparent',
                        transition: 'background 0.15s ease',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent-blue)' }}>
                          {job.job_id.length > 16 ? `${job.job_id.substring(0, 16)}...` : job.job_id}
                        </span>
                        <ReplayStatusBadge status={job.status} />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.73rem', color: 'var(--text-secondary)' }}>
                        <span>
                          {job.processed_records} / {job.total_records} records
                        </span>
                        <span>{new Date(job.created_at).toLocaleTimeString()}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ padding: '2rem 1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                No replay jobs found.
              </div>
            )}
          </div>
        </div>

        {/* Right Column (2 spans): Selected Job Details & Observability */}
        <div style={{ gridColumn: 'span 2' }}>
          {selectedJob ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              {/* Header card */}
              <div className="card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      <span style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                        {selectedJob.job_id}
                      </span>
                      <ReplayStatusBadge status={selectedJob.status} />
                      <span className="badge badge-neutral">{selectedJob.mode}</span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                      Created at {new Date(selectedJob.created_at).toLocaleString()}
                      {selectedJob.idempotency_key && ` • Idempotency: ${selectedJob.idempotency_key}`}
                    </div>
                  </div>

                  {/* Cancellation Button with Confirmation Guard */}
                  <div>
                    {isSelectedCancellable ? (
                      confirmingCancel ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                          <span style={{ fontSize: '0.72rem', color: 'var(--accent-red)', fontWeight: 600 }}>
                            Confirm abort?
                          </span>
                          <button
                            className="btn btn-danger"
                            onClick={() => {
                              setConfirmingCancel(false);
                              handleCancelJob(selectedJob.job_id);
                            }}
                            disabled={isCancelling}
                            style={{ padding: '0.35rem 0.65rem', fontSize: '0.72rem' }}
                          >
                            <span>Confirm Cancel</span>
                          </button>
                          <button
                            className="btn btn-secondary"
                            onClick={() => setConfirmingCancel(false)}
                            style={{ padding: '0.35rem 0.55rem', fontSize: '0.72rem' }}
                          >
                            <span>Keep Running</span>
                          </button>
                        </div>
                      ) : (
                        <button
                          className="btn btn-danger"
                          onClick={() => setConfirmingCancel(true)}
                          disabled={isCancelling}
                        >
                          <XCircle size={14} />
                          <span>Cancel Replay Job</span>
                        </button>
                      )
                    ) : (
                      <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                        Job in terminal state ({selectedJob.status})
                      </span>
                    )}
                  </div>
                </div>

                {cancelMessage && (
                  <div
                    style={{
                      marginTop: '0.75rem',
                      padding: '0.6rem 0.85rem',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '0.78rem',
                      background:
                        cancelMessage.type === 'success'
                          ? 'rgba(16, 185, 129, 0.1)'
                          : 'rgba(239, 68, 68, 0.1)',
                      color:
                        cancelMessage.type === 'success'
                          ? 'var(--accent-green)'
                          : 'var(--accent-red)',
                      border: `1px solid ${
                        cancelMessage.type === 'success'
                          ? 'rgba(16, 185, 129, 0.3)'
                          : 'rgba(239, 68, 68, 0.3)'
                      }`,
                    }}
                  >
                    {cancelMessage.text}
                  </div>
                )}

                {/* Progress bar */}
                <div style={{ marginTop: '1.25rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.35rem' }}>
                    <span>
                      Execution Progress ({selectedJob.processed_records} / {selectedJob.total_records} records)
                    </span>
                    <span style={{ fontWeight: 600 }}>
                      {selectedJob.progress?.progress_percentage
                        ? `${selectedJob.progress.progress_percentage.toFixed(1)}%`
                        : `${((selectedJob.processed_records / (selectedJob.total_records || 1)) * 100).toFixed(1)}%`}
                    </span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${
                          selectedJob.progress?.progress_percentage ??
                          (selectedJob.processed_records / (selectedJob.total_records || 1)) * 100
                        }%`,
                        background:
                          selectedJob.status === 'COMPLETED'
                            ? 'var(--accent-green)'
                            : selectedJob.status === 'FAILED'
                            ? 'var(--accent-red)'
                            : selectedJob.status === 'CANCELLED'
                            ? 'var(--text-muted)'
                            : 'var(--accent-blue)',
                      }}
                    />
                  </div>
                </div>

                {/* Live execution telemetry */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem', marginTop: '1rem' }}>
                  <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Processed</div>
                    <div style={{ fontSize: '1rem', fontWeight: 700, marginTop: '2px' }}>{selectedJob.processed_records}</div>
                  </div>
                  <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Failed / Invalid</div>
                    <div style={{ fontSize: '1rem', fontWeight: 700, color: selectedJob.failed_records > 0 ? 'var(--accent-red)' : 'var(--text-primary)', marginTop: '2px' }}>
                      {selectedJob.failed_records}
                    </div>
                  </div>
                  <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Throughput</div>
                    <div style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '2px' }}>
                      {selectedJob.progress?.throughput_records_per_sec !== null && selectedJob.progress?.throughput_records_per_sec !== undefined
                        ? `${selectedJob.progress.throughput_records_per_sec.toFixed(1)} rec/s`
                        : 'N/A'}
                    </div>
                  </div>
                  <div style={{ background: 'var(--bg-tertiary)', padding: '0.65rem', borderRadius: 'var(--radius-sm)' }}>
                    <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Elapsed Time</div>
                    <div style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '2px' }}>
                      {selectedJob.progress?.elapsed_duration_sec !== undefined
                        ? `${selectedJob.progress.elapsed_duration_sec.toFixed(2)} s`
                        : selectedJob.metrics_summary?.latency_ms?.total_duration_sec !== undefined
                        ? `${selectedJob.metrics_summary.latency_ms.total_duration_sec.toFixed(2)} s`
                        : 'N/A'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Metrics Summary if available */}
              {selectedJob.metrics_summary ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  {/* Financial Exposure & Latency */}
                  <div className="grid grid-cols-2">
                    {/* Financial Exposure Card */}
                    <div className="card">
                      <div className="card-header">
                        <div className="card-title">Financial Exposure</div>
                        <div className="card-subtitle">Volume replayed vs. flagged amounts</div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Total Amount Evaluated:</span>
                          <strong style={{ fontFamily: 'var(--font-mono)' }}>
                            ${selectedJob.metrics_summary.financial_exposure.total_amount_replayed.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                          </strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Flagged Fraud Exposure:</span>
                          <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-red)' }}>
                            ${selectedJob.metrics_summary.financial_exposure.flagged_amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                          </strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Flagged Amount Percentage:</span>
                          <strong style={{ fontFamily: 'var(--font-mono)' }}>
                            {selectedJob.metrics_summary.financial_exposure.flagged_amount_percentage.toFixed(2)}%
                          </strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                          <span style={{ color: 'var(--text-muted)' }}>Flagged Record Percentage:</span>
                          <strong style={{ fontFamily: 'var(--font-mono)' }}>
                            {selectedJob.metrics_summary.financial_exposure.flagged_record_percentage.toFixed(2)}%
                          </strong>
                        </div>
                      </div>
                    </div>

                    {/* Latency Distribution */}
                    <div className="card">
                      <div className="card-header">
                        <div className="card-title">Latency Telemetry (ms)</div>
                        <div className="card-subtitle">Inference latency percentiles across batch</div>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', textAlign: 'center' }}>
                        <div style={{ background: 'var(--bg-tertiary)', padding: '0.5rem', borderRadius: 'var(--radius-sm)' }}>
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>MIN</div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 600, marginTop: '2px' }}>
                            {selectedJob.metrics_summary.latency_ms.min.toFixed(1)} ms
                          </div>
                        </div>
                        <div style={{ background: 'var(--bg-tertiary)', padding: '0.5rem', borderRadius: 'var(--radius-sm)' }}>
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>MEAN</div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 600, marginTop: '2px' }}>
                            {selectedJob.metrics_summary.latency_ms.mean.toFixed(1)} ms
                          </div>
                        </div>
                        <div style={{ background: 'var(--bg-tertiary)', padding: '0.5rem', borderRadius: 'var(--radius-sm)' }}>
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>P50 (MEDIAN)</div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 600, marginTop: '2px' }}>
                            {selectedJob.metrics_summary.latency_ms.p50.toFixed(1)} ms
                          </div>
                        </div>
                        <div style={{ background: 'var(--bg-tertiary)', padding: '0.5rem', borderRadius: 'var(--radius-sm)' }}>
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>P95</div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-amber)', marginTop: '2px' }}>
                            {selectedJob.metrics_summary.latency_ms.p95.toFixed(1)} ms
                          </div>
                        </div>
                        <div style={{ background: 'var(--bg-tertiary)', padding: '0.5rem', borderRadius: 'var(--radius-sm)' }}>
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>MAX</div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 600, marginTop: '2px' }}>
                            {selectedJob.metrics_summary.latency_ms.max.toFixed(1)} ms
                          </div>
                        </div>
                        <div style={{ background: 'var(--bg-tertiary)', padding: '0.5rem', borderRadius: 'var(--radius-sm)' }}>
                          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>TOTAL DURATION</div>
                          <div style={{ fontSize: '0.85rem', fontWeight: 600, marginTop: '2px' }}>
                            {selectedJob.metrics_summary.latency_ms.total_duration_sec.toFixed(2)} s
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Risk Tier & Action Counts */}
                  <div className="grid grid-cols-2">
                    <div className="card">
                      <div className="card-header">
                        <div className="card-title">Risk Tier Distribution</div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                        {Object.entries(selectedJob.metrics_summary.risk_tier_counts || {}).map(([tier, count]) => (
                          <div key={tier} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                            <span>{tier}</span>
                            <strong>{count}</strong>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="card">
                      <div className="card-header">
                        <div className="card-title">Recommended Actions Count</div>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                        {Object.entries(selectedJob.metrics_summary.action_counts || {}).map(([act, count]) => (
                          <div key={act} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem' }}>
                            <span>{act}</span>
                            <strong>{count}</strong>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Zero Side-Effect Ledger Verification Box */}
                  <div
                    style={{
                      background: 'rgba(16, 185, 129, 0.06)',
                      border: '1px solid rgba(16, 185, 129, 0.25)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '0.85rem 1rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      fontSize: '0.78rem',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-green)' }}>
                      <CheckCircle2 size={16} />
                      <strong style={{ color: 'var(--text-primary)' }}>Non-Interference Verification:</strong>
                    </div>
                    <div>
                      Alerts Created: <strong>{selectedJob.metrics_summary.state_mutations.alerts_created}</strong> • Settlements Applied: <strong>{selectedJob.metrics_summary.state_mutations.settlements_applied}</strong>
                    </div>
                  </div>
                </div>
              ) : selectedJob.status === 'RUNNING' || selectedJob.status === 'PENDING' ? (
                <div className="card" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                  <Activity size={28} style={{ animation: 'spin 2s linear infinite', margin: '0 auto 0.75rem auto' }} />
                  <div>Job in progress... Final metrics will compile once simulation completes.</div>
                </div>
              ) : (
                <div className="card" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                  Detailed metrics unavailable for this job.
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
                padding: '5rem 2rem',
                textAlign: 'center',
                color: 'var(--text-muted)',
                gap: '1rem',
              }}
            >
              <History size={40} />
              <div style={{ fontSize: '0.95rem', fontWeight: 600 }}>Select a Replay Job</div>
              <div style={{ fontSize: '0.82rem', maxWidth: '340px' }}>
                Select a completed or running job on the left to inspect latency percentiles, exposure calculations, and non-interference metrics.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
