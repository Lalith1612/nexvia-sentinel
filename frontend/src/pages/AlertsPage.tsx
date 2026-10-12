import React, { useEffect, useState } from 'react';
import {
  ShieldAlert,
  Search,
  Filter,
  RefreshCw,
  Eye,
  ChevronLeft,
  ChevronRight,
  SlidersHorizontal,
} from 'lucide-react';
import { api } from '../api/client';
import { AlertFilters, AlertListResponse, AlertResponse, AlertStatus, RiskTier } from '../types';
import { RiskTierBadge, ActionBadge, AlertStatusBadge } from '../components/Badge';
import { LoadingState, ErrorAlert } from '../components/Feedback';
import { AlertDetailModal } from '../components/AlertDetailModal';

interface AlertsPageProps {
  onRefreshTrigger?: number;
}

export const AlertsPage: React.FC<AlertsPageProps> = ({ onRefreshTrigger }) => {
  const [alerts, setAlerts] = useState<AlertResponse[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [limit] = useState<number>(20);
  const [offset, setOffset] = useState<number>(0);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [tierFilter, setTierFilter] = useState<string>('ALL');
  const [minScoreFilter, setMinScoreFilter] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Selected Alert for inspection modal
  const [selectedAlert, setSelectedAlert] = useState<AlertResponse | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAlerts = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const filters: AlertFilters = {
        limit,
        offset,
      };

      if (statusFilter !== 'ALL') {
        filters.status = statusFilter as AlertStatus;
      }
      if (tierFilter !== 'ALL') {
        filters.risk_tier = tierFilter as RiskTier;
      }
      if (minScoreFilter > 0) {
        filters.min_score = minScoreFilter;
      }

      const res = await api.getAlerts(filters);
      setAlerts(res.items);
      setTotal(res.total);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch alerts from Sentinel ledger');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
  }, [offset, statusFilter, tierFilter, minScoreFilter, onRefreshTrigger]);

  const handleAlertUpdated = (updated: AlertResponse) => {
    setAlerts((prev) =>
      prev.map((a) => (a.alert_id === updated.alert_id ? updated : a))
    );
    setSelectedAlert(updated);
  };

  // Client-side text search by ID or account name
  const filteredAlerts = alerts.filter((a) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      a.alert_id.toLowerCase().includes(q) ||
      a.transaction_id.toLowerCase().includes(q) ||
      a.name_orig.toLowerCase().includes(q) ||
      a.name_dest.toLowerCase().includes(q)
    );
  });

  const totalPages = Math.ceil(total / limit) || 1;
  const currentPage = Math.floor(offset / limit) + 1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Filter and Control Bar */}
      <div className="card" style={{ padding: '1rem' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center', justifyContent: 'space-between' }}>
          {/* Search Input */}
          <div style={{ position: 'relative', minWidth: '260px', flex: '1 1 260px' }}>
            <Search
              size={15}
              style={{
                position: 'absolute',
                left: '10px',
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--text-muted)',
              }}
            />
            <input
              type="text"
              className="form-control"
              style={{ paddingLeft: '32px' }}
              placeholder="Search Alert ID, Tx ID, or Account..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Filter Dropdowns */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
            {/* Status Filter */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Status:</span>
              <select
                className="form-control"
                style={{ width: 'auto', padding: '0.4rem 0.6rem', fontSize: '0.78rem' }}
                value={statusFilter}
                onChange={(e) => {
                  setOffset(0);
                  setStatusFilter(e.target.value);
                }}
              >
                <option value="ALL">All Statuses</option>
                <option value="OPEN">Open</option>
                <option value="INVESTIGATING">Investigating</option>
                <option value="CONFIRMED_FRAUD">Confirmed Fraud</option>
                <option value="FALSE_POSITIVE">False Positive</option>
                <option value="CLOSED">Closed</option>
              </select>
            </div>

            {/* Risk Tier Filter */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Tier:</span>
              <select
                className="form-control"
                style={{ width: 'auto', padding: '0.4rem 0.6rem', fontSize: '0.78rem' }}
                value={tierFilter}
                onChange={(e) => {
                  setOffset(0);
                  setTierFilter(e.target.value);
                }}
              >
                <option value="ALL">All Tiers</option>
                <option value="CRITICAL">Critical</option>
                <option value="HIGH">High</option>
                <option value="MEDIUM">Medium</option>
                <option value="LOW">Low</option>
              </select>
            </div>

            {/* Min Score Filter */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Min Score:</span>
              <select
                className="form-control"
                style={{ width: 'auto', padding: '0.4rem 0.6rem', fontSize: '0.78rem' }}
                value={minScoreFilter}
                onChange={(e) => {
                  setOffset(0);
                  setMinScoreFilter(parseFloat(e.target.value));
                }}
              >
                <option value="0">0.00+</option>
                <option value="0.5">0.50+</option>
                <option value="0.7">0.70+</option>
                <option value="0.85">0.85+ (High/Crit)</option>
              </select>
            </div>

            <button
              className="btn btn-secondary"
              onClick={fetchAlerts}
              disabled={isLoading}
              style={{ padding: '0.4rem 0.6rem' }}
              title="Refresh alerts"
            >
              <RefreshCw size={14} style={{ animation: isLoading ? 'spin 1s linear infinite' : 'none' }} />
            </button>
          </div>
        </div>
      </div>

      {/* Error state */}
      {error && <ErrorAlert message={error} onRetry={fetchAlerts} />}

      {/* Alerts Table */}
      <div className="card" style={{ padding: 0 }}>
        {isLoading && alerts.length === 0 ? (
          <LoadingState message="Loading alert records from Sentinel database..." />
        ) : filteredAlerts.length > 0 ? (
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Alert ID</th>
                  <th>Step</th>
                  <th>Type</th>
                  <th>Amount</th>
                  <th>Origin Account</th>
                  <th>Dest Account</th>
                  <th>Score</th>
                  <th>Tier</th>
                  <th>Status</th>
                  <th>Recommended Action</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredAlerts.map((alert) => (
                  <tr
                    key={alert.alert_id}
                    onClick={() => setSelectedAlert(alert)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--accent-blue)' }}>
                      {alert.alert_id.length > 16 ? `${alert.alert_id.substring(0, 16)}...` : alert.alert_id}
                    </td>
                    <td style={{ fontSize: '0.8rem' }}>{alert.step}</td>
                    <td>
                      <span className="badge badge-neutral">{alert.type}</span>
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', fontWeight: 600 }}>
                      ${alert.amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                      {alert.name_orig}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                      {alert.name_dest}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', fontWeight: 700 }}>
                      {alert.hybrid_risk_score.toFixed(4)}
                    </td>
                    <td>
                      <RiskTierBadge tier={alert.risk_tier} />
                    </td>
                    <td>
                      <AlertStatusBadge status={alert.status} />
                    </td>
                    <td>
                      <ActionBadge action={alert.recommended_action} />
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-secondary"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedAlert(alert);
                        }}
                        style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                      >
                        <Eye size={12} />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div
            style={{
              padding: '2.5rem 1rem',
              textAlign: 'center',
              color: 'var(--text-muted)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <ShieldAlert size={28} color="var(--border-accent)" />
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              No alerts matching the selected filters.
            </div>
            <div style={{ fontSize: '0.74rem' }}>
              Try adjusting your search criteria, minimum score, or risk tier filter.
            </div>
          </div>
        )}

        {/* Pagination Bar */}
        <div
          style={{
            padding: '0.75rem 1rem',
            borderTop: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '0.78rem',
            color: 'var(--text-secondary)',
          }}
        >
          <div>
            Showing {filteredAlerts.length} of {total} alerts recorded
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>
              Page {currentPage} of {totalPages}
            </span>
            <button
              className="btn btn-secondary"
              disabled={offset === 0 || isLoading}
              onClick={() => setOffset((prev) => Math.max(0, prev - limit))}
              style={{ padding: '0.25rem 0.5rem' }}
              aria-label="Previous page"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              className="btn btn-secondary"
              disabled={offset + limit >= total || isLoading}
              onClick={() => setOffset((prev) => prev + limit)}
              style={{ padding: '0.25rem 0.5rem' }}
              aria-label="Next page"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Alert Details Modal */}
      {selectedAlert && (
        <AlertDetailModal
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
          onAlertUpdated={handleAlertUpdated}
        />
      )}
    </div>
  );
};
