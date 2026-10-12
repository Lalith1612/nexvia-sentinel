import React from 'react';
import {
  LayoutDashboard,
  ShieldAlert,
  Binary,
  History,
  Radio,
  Server,
  Layers,
  ExternalLink,
} from 'lucide-react';
import { ServiceHealthStatus } from '../types';

export type NavTab = 'overview' | 'alerts' | 'prediction' | 'replay' | 'streaming';

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  systemStatus: ServiceHealthStatus | 'CONNECTING' | 'ERROR';
  openAlertsCount?: number;
  isOpenMobile?: boolean;
  onCloseMobile?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  systemStatus,
  openAlertsCount,
  isOpenMobile = false,
  onCloseMobile,
}) => {
  const navItems = [
    {
      id: 'overview' as NavTab,
      label: 'Overview',
      icon: LayoutDashboard,
      badge: null,
    },
    {
      id: 'alerts' as NavTab,
      label: 'Alert Investigation',
      icon: ShieldAlert,
      badge: openAlertsCount !== undefined && openAlertsCount > 0 ? openAlertsCount : null,
    },
    {
      id: 'prediction' as NavTab,
      label: 'Fraud Prediction',
      icon: Binary,
      badge: null,
    },
    {
      id: 'replay' as NavTab,
      label: 'Replay Monitor',
      icon: History,
      badge: null,
    },
    {
      id: 'streaming' as NavTab,
      label: 'Streaming Status',
      icon: Radio,
      badge: 'MOCK',
    },
  ];

  return (
    <aside className={`sidebar ${isOpenMobile ? 'open' : ''}`}>
      {/* Nexvia Official Brand Header */}
      <div className="sidebar-brand">
        <div className="brand-logo-row">
          <img
            src="/nexvia-logo.png"
            alt="Nexvia Technologies"
            className="brand-logo-img"
          />
        </div>

        {/* Unified Sentinel Product Lockup */}
        <div className="brand-product-lockup">
          <div className="brand-product-title">
            <span className="dot" />
            <span>SENTINEL</span>
          </div>
          <span className="brand-product-sub">FRAUD INTELLIGENCE</span>
        </div>
      </div>

      {/* Main Navigation */}
      <nav className="sidebar-nav" role="tablist">
        <div className="nav-section-label">Operations Console</div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => {
                onSelectTab(item.id);
                if (onCloseMobile) onCloseMobile();
              }}
              className={`nav-item ${isActive ? 'active' : ''}`}
              role="tab"
              aria-selected={isActive}
            >
              <Icon size={17} />
              <span style={{ flex: 1 }}>{item.label}</span>
              {item.badge !== null && (
                <span
                  className="nav-badge"
                  style={{
                    background:
                      typeof item.badge === 'number'
                        ? 'var(--badge-danger-bg)'
                        : 'rgba(255, 255, 255, 0.05)',
                    color:
                      typeof item.badge === 'number'
                        ? 'var(--accent-red)'
                        : 'var(--text-muted)',
                    border:
                      typeof item.badge === 'number'
                        ? '1px solid var(--badge-danger-border)'
                        : '1px solid var(--border-subtle)',
                  }}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Backend Status & Nexvia Technologies Attribution */}
      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.45rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
            <Server size={13} color="var(--accent-cyan)" />
            <span>FastAPI Core</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.74rem', fontWeight: 600 }}>
            <span
              style={{
                width: '7px',
                height: '7px',
                borderRadius: '50%',
                background:
                  systemStatus === 'HEALTHY'
                    ? 'var(--accent-green)'
                    : systemStatus === 'DEGRADED'
                    ? 'var(--accent-amber)'
                    : 'var(--accent-red)',
                display: 'inline-block',
                boxShadow:
                  systemStatus === 'HEALTHY'
                    ? '0 0 8px rgba(16, 185, 129, 0.6)'
                    : 'none',
              }}
            />
            <span
              style={{
                color:
                  systemStatus === 'HEALTHY'
                    ? 'var(--accent-green)'
                    : systemStatus === 'DEGRADED'
                    ? 'var(--accent-amber)'
                    : 'var(--accent-red)',
                fontFamily: 'var(--font-mono)',
              }}
            >
              {systemStatus}
            </span>
          </div>
        </div>

        <div className="footer-attribution">
          <span>NEXVIA TECHNOLOGIES</span>
          <span style={{ fontFamily: 'var(--font-mono)' }}>:8000</span>
        </div>
      </div>
    </aside>
  );
};
