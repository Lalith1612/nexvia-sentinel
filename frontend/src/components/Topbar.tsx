import React from 'react';
import { RefreshCw, AlertCircle, Menu, Layers } from 'lucide-react';
import { NavTab } from './Sidebar';

interface TopbarProps {
  currentTab: NavTab;
  onRefresh: () => void;
  isLoading: boolean;
  lastUpdated: Date | null;
  onToggleMobileMenu?: () => void;
}

export const Topbar: React.FC<TopbarProps> = ({
  currentTab,
  onRefresh,
  isLoading,
  lastUpdated,
  onToggleMobileMenu,
}) => {
  const titles: Record<NavTab, { title: string; subtitle: string }> = {
    overview: {
      title: 'Operations Overview',
      subtitle: 'System health, model artifacts, and real-time risk posture',
    },
    alerts: {
      title: 'Alert Investigation & Triage',
      subtitle: 'Review flagged high-risk transactions and record analyst actions',
    },
    prediction: {
      title: 'Pre-Authorization Inference',
      subtitle: 'Evaluate transactions against dual tabular & relational ensemble',
    },
    replay: {
      title: 'Bounded Transaction Replay',
      subtitle: 'Simulate historical test slices under dry-run non-interference',
    },
    streaming: {
      title: 'Streaming Infrastructure',
      subtitle: 'Kafka consumer topology and real-time event pipeline status',
    },
  };

  const { title } = titles[currentTab] || {
    title: 'Sentinel Console',
    subtitle: 'Financial Fraud Platform',
  };

  return (
    <header className="topbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
        {onToggleMobileMenu && (
          <button
            className="btn btn-secondary"
            onClick={onToggleMobileMenu}
            style={{ padding: '0.35rem 0.5rem', display: 'flex' }}
            aria-label="Toggle Navigation"
          >
            <Menu size={16} />
          </button>
        )}
        <div>
          <div className="topbar-branding-crumb">
            <span>NEXVIA</span>
            <span>/</span>
            <strong>SENTINEL</strong>
          </div>
          <div className="topbar-title">{title}</div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.45rem',
            padding: '0.25rem 0.65rem',
            borderRadius: 'var(--radius-sm)',
            background: 'rgba(245, 158, 11, 0.08)',
            border: '1px solid rgba(245, 158, 11, 0.25)',
            fontSize: '0.72rem',
            color: 'var(--accent-amber)',
          }}
        >
          <AlertCircle size={12} />
          <span style={{ fontWeight: 600, letterSpacing: '0.02em' }}>LOCAL / MOCK BROKER</span>
        </div>

        {lastUpdated && (
          <div
            style={{
              fontSize: '0.72rem',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.3rem',
            }}
          >
            <span>SYNCD:</span>
            <span style={{ color: 'var(--text-secondary)' }}>
              {lastUpdated.toLocaleTimeString()}
            </span>
          </div>
        )}

        <button
          className="btn btn-secondary"
          onClick={onRefresh}
          disabled={isLoading}
          aria-label="Refresh view data"
          style={{ padding: '0.4rem 0.75rem', fontSize: '0.75rem' }}
        >
          <RefreshCw
            size={13}
            style={{
              animation: isLoading ? 'spin 0.8s linear infinite' : 'none',
              color: 'var(--accent-cyan)',
            }}
          />
          <span>Sync</span>
        </button>
      </div>
    </header>
  );
};
