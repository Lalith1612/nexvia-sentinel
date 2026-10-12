import React from 'react';
import { AlertStatus, RecommendedAction, RiskTier, ServiceHealthStatus, ReplayJobStatus } from '../types';

interface BadgeProps {
  children?: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'purple' | 'neutral';
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({ children, variant = 'neutral', className = '' }) => {
  return (
    <span className={`badge badge-${variant} ${className}`}>
      {children}
    </span>
  );
};

export const HealthBadge: React.FC<{ status: ServiceHealthStatus }> = ({ status }) => {
  switch (status) {
    case 'HEALTHY':
      return <Badge variant="success">HEALTHY</Badge>;
    case 'DEGRADED':
      return <Badge variant="warning">DEGRADED</Badge>;
    case 'UNHEALTHY':
      return <Badge variant="danger">UNHEALTHY</Badge>;
    default:
      return <Badge variant="neutral">{status}</Badge>;
  }
};

export const RiskTierBadge: React.FC<{ tier: RiskTier }> = ({ tier }) => {
  switch (tier) {
    case 'CRITICAL':
      return <Badge variant="danger">CRITICAL</Badge>;
    case 'HIGH':
      return <Badge variant="warning">HIGH</Badge>;
    case 'MEDIUM':
      return <Badge variant="info">MEDIUM</Badge>;
    case 'LOW':
      return <Badge variant="success">LOW</Badge>;
    default:
      return <Badge variant="neutral">{tier}</Badge>;
  }
};

export const ActionBadge: React.FC<{ action: RecommendedAction }> = ({ action }) => {
  switch (action) {
    case 'AUTOMATED_BLOCK':
      return <Badge variant="danger">AUTOMATED BLOCK</Badge>;
    case 'STEP_UP_AUTHENTICATION':
      return <Badge variant="warning">STEP-UP AUTH</Badge>;
    case 'MANUAL_ANALYST_REVIEW':
      return <Badge variant="info">MANUAL REVIEW</Badge>;
    case 'AUTOMATED_CLEAR_AND_SETTLE':
      return <Badge variant="success">CLEAR & SETTLE</Badge>;
    default:
      return <Badge variant="neutral">{action}</Badge>;
  }
};

export const AlertStatusBadge: React.FC<{ status: AlertStatus }> = ({ status }) => {
  switch (status) {
    case 'OPEN':
      return <Badge variant="danger">OPEN</Badge>;
    case 'INVESTIGATING':
      return <Badge variant="warning">INVESTIGATING</Badge>;
    case 'CONFIRMED_FRAUD':
      return <Badge variant="purple">CONFIRMED FRAUD</Badge>;
    case 'FALSE_POSITIVE':
      return <Badge variant="info">FALSE POSITIVE</Badge>;
    case 'CLOSED':
      return <Badge variant="neutral">CLOSED</Badge>;
    default:
      return <Badge variant="neutral">{status}</Badge>;
  }
};

export const ReplayStatusBadge: React.FC<{ status: ReplayJobStatus }> = ({ status }) => {
  switch (status) {
    case 'COMPLETED':
      return <Badge variant="success">COMPLETED</Badge>;
    case 'RUNNING':
      return <Badge variant="info">RUNNING</Badge>;
    case 'PENDING':
      return <Badge variant="warning">PENDING</Badge>;
    case 'FAILED':
      return <Badge variant="danger">FAILED</Badge>;
    case 'CANCELLED':
      return <Badge variant="neutral">CANCELLED</Badge>;
    default:
      return <Badge variant="neutral">{status}</Badge>;
  }
};
