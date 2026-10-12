import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ message = 'Loading data from Sentinel backend...' }) => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '4rem 2rem',
        color: 'var(--text-secondary)',
        gap: '1rem',
      }}
    >
      <div className="spinner" />
      <div style={{ fontSize: '0.88rem' }}>{message}</div>
    </div>
  );
};

interface ErrorAlertProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  title = 'Connection or Request Error',
  message,
  onRetry,
}) => {
  return (
    <div
      style={{
        background: 'rgba(239, 68, 68, 0.08)',
        border: '1px solid rgba(239, 68, 68, 0.3)',
        borderRadius: 'var(--radius-md)',
        padding: '1.25rem',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.75rem',
        margin: '1rem 0',
      }}
      role="alert"
    >
      <AlertTriangle size={18} style={{ color: 'var(--accent-red)', flexShrink: 0, marginTop: '2px' }} />
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 600, color: 'var(--accent-red)', fontSize: '0.88rem', marginBottom: '0.25rem' }}>
          {title}
        </div>
        <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', wordBreak: 'break-word' }}>
          {message}
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="btn btn-secondary"
            style={{ marginTop: '0.75rem', padding: '0.35rem 0.75rem', fontSize: '0.78rem' }}
          >
            <RefreshCw size={13} />
            <span>Retry Request</span>
          </button>
        )}
      </div>
    </div>
  );
};
