import React from 'react';
import {
  Radio,
  AlertTriangle,
  Server,
  Layers,
  ArrowRight,
  Shield,
  Cpu,
  Info,
  CheckCircle2,
  XCircle,
} from 'lucide-react';

export const StreamingPage: React.FC = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Prominent Truth-in-Reporting Warning Banner */}
      <div
        style={{
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.35)',
          borderRadius: 'var(--radius-md)',
          padding: '1.25rem',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '1rem',
        }}
      >
        <AlertTriangle size={24} style={{ color: 'var(--accent-amber)', flexShrink: 0, marginTop: '2px' }} />
        <div>
          <div style={{ fontWeight: 700, color: 'var(--accent-amber)', fontSize: '0.95rem', marginBottom: '0.25rem' }}>
            Real Kafka Broker Connectivity: NOT VERIFIED
          </div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Docker and live distributed message brokers are unavailable in this development environment.
            The Sentinel streaming ingestion layer was built with full partition and consumer lifecycle contracts,
            and tested extensively under mocked broker fixtures. However, end-to-end integration against a live Kafka cluster
            has not been executed and must not be reported as operational.
          </div>
        </div>
      </div>

      {/* Streaming Pipeline Architecture Overview */}
      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Real-Time Event Streaming Topology</div>
            <div className="card-subtitle">Architectural design for low-latency transaction ingestion</div>
          </div>
          <span className="badge badge-warning">Design & Local Mock</span>
        </div>

        {/* Visual Pipeline Flow */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '1rem',
            padding: '1rem 0',
            alignItems: 'center',
          }}
        >
          {/* Step 1: Upstream Gateway */}
          <div
            style={{
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Source</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, margin: '0.4rem 0' }}>Payment Gateway</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>External REST / RPC Ingest</div>
            <div style={{ marginTop: '0.75rem' }}>
              <span className="badge badge-info">JSON Payload</span>
            </div>
          </div>

          {/* Step 2: Kafka Broker */}
          <div
            style={{
              background: 'var(--bg-tertiary)',
              border: '1px solid rgba(245, 158, 11, 0.4)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '0.7rem', color: 'var(--accent-amber)', textTransform: 'uppercase', fontWeight: 600 }}>Message Bus</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, margin: '0.4rem 0' }}>Kafka Broker</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Topic: sentinel.transactions</div>
            <div style={{ marginTop: '0.75rem' }}>
              <span className="badge badge-warning">Not Verified</span>
            </div>
          </div>

          {/* Step 3: Consumer Engine */}
          <div
            style={{
              background: 'var(--bg-tertiary)',
              border: '1px solid var(--border-color)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Worker</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, margin: '0.4rem 0' }}>Sentinel Consumer</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Partition Key: nameOrig</div>
            <div style={{ marginTop: '0.75rem' }}>
              <span className="badge badge-neutral">Mock Validated</span>
            </div>
          </div>

          {/* Step 4: Inference & Ledger */}
          <div
            style={{
              background: 'var(--bg-tertiary)',
              border: '1px solid rgba(16, 185, 129, 0.4)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '0.7rem', color: 'var(--accent-green)', textTransform: 'uppercase', fontWeight: 600 }}>Destination</div>
            <div style={{ fontSize: '0.95rem', fontWeight: 700, margin: '0.4rem 0' }}>Dual Ensemble & DB</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>XGBoost + GraphSAGE</div>
            <div style={{ marginTop: '0.75rem' }}>
              <span className="badge badge-success">Production Ready</span>
            </div>
          </div>
        </div>
      </div>

      {/* Configuration & Specifications Grid */}
      <div className="grid grid-cols-2">
        {/* Streaming Ingestion Settings */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Consumer Contract Specification</div>
            <div className="card-subtitle">Parameters defined in Sentinel streaming architecture</div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.82rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Target Topic:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>sentinel.transactions</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Consumer Group ID:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>sentinel-fraud-analysts</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Partition Key Strategy:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>nameOrig (Sender Account)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Auto Offset Reset:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>earliest</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Offset Commit Strategy:</span>
              <strong style={{ color: 'var(--accent-blue)' }}>Manual post-settlement commit</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0' }}>
              <span style={{ color: 'var(--text-muted)' }}>Idempotency Guard:</span>
              <strong style={{ color: 'var(--accent-green)' }}>SQLite Unique(transaction_id)</strong>
            </div>
          </div>
        </div>

        {/* Technical Caveats & Pre-Requisites */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Deployment Prerequisites</div>
            <div className="card-subtitle">Required infrastructure for production deployment</div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <CheckCircle2 size={15} color="var(--accent-green)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>Strict In-Order Evaluation:</strong> Partitioning by <code>nameOrig</code> ensures that transactions from the same account never evaluate out of chronological order.
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <CheckCircle2 size={15} color="var(--accent-green)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>At-Least-Once Delivery Resilience:</strong> Consumer only commits Kafka offset after transaction scoring and ledger writes succeed.
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <XCircle size={15} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>Live Cluster Provisioning:</strong> Requires Apache Kafka ≥ 3.0 cluster with ZooKeeper / KRaft, unreachable in local Windows desktop environment.
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
              <Info size={15} color="var(--accent-blue)" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>Future Verification:</strong> Once containerized with Docker Compose in a cloud environment, end-to-end load tests should verify cluster throughput.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
