import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { AlertsPage } from '../pages/AlertsPage';
import { api } from '../api/client';
import { AlertResponse } from '../types';

vi.mock('../api/client', () => ({
  api: {
    getAlerts: vi.fn(),
    updateAlert: vi.fn(),
  },
}));

const mockAlert: AlertResponse = {
  alert_id: 'ALT_998877665544',
  transaction_id: 'TX_11223344',
  step: 622,
  type: 'TRANSFER',
  amount: 250000,
  name_orig: 'C10001',
  name_dest: 'C20002',
  hybrid_risk_score: 0.9421,
  risk_tier: 'CRITICAL',
  recommended_action: 'AUTOMATED_BLOCK',
  status: 'OPEN',
  analyst_id: null,
  resolution_notes: null,
  created_at: '2026-10-12T00:00:00Z',
  updated_at: '2026-10-12T00:00:00Z',
};

describe('AlertsPage & AlertDetailModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders alerts table correctly with data from backend', async () => {
    (api.getAlerts as any).mockResolvedValue({
      total: 1,
      limit: 20,
      offset: 0,
      items: [mockAlert],
    });

    render(<AlertsPage />);

    await waitFor(() => {
      expect(screen.getByText('ALT_998877665544')).toBeInTheDocument();
      expect(screen.getByText('$250,000.00')).toBeInTheDocument();
      expect(screen.getByText('0.9421')).toBeInTheDocument();
      expect(screen.getByText('CRITICAL')).toBeInTheDocument();
    });
  });

  it('opens alert modal on inspect and allows triage progression', async () => {
    (api.getAlerts as any).mockResolvedValue({
      total: 1,
      limit: 20,
      offset: 0,
      items: [mockAlert],
    });

    render(<AlertsPage />);

    await waitFor(() => {
      expect(screen.getByText('ALT_998877665544')).toBeInTheDocument();
    });

    // Click inspect
    fireEvent.click(screen.getByText('Inspect'));

    // Modal should appear
    expect(screen.getByText('Alert Investigation')).toBeInTheDocument();
    expect(screen.getByText('Model Advisory Note:')).toBeInTheDocument();

    // Fill analyst ID
    const analystInput = screen.getByPlaceholderText(/analyst_42/i);
    fireEvent.change(analystInput, { target: { value: 'analyst_bob' } });

    (api.updateAlert as any).mockResolvedValue({
      ...mockAlert,
      status: 'INVESTIGATING',
      analyst_id: 'analyst_bob',
    });

    // Commit triage update
    fireEvent.click(screen.getByText('Commit Triage Update'));

    await waitFor(() => {
      expect(api.updateAlert).toHaveBeenCalledWith('ALT_998877665544', {
        status: 'INVESTIGATING',
        analyst_id: 'analyst_bob',
        resolution_notes: null,
      });
    });
  });

  it('shows terminal state banner when alert is CONFIRMED_FRAUD', async () => {
    const terminalAlert: AlertResponse = {
      ...mockAlert,
      status: 'CONFIRMED_FRAUD',
      analyst_id: 'analyst_bob',
      resolution_notes: 'Confirmed unauthorized account takeover.',
    };

    (api.getAlerts as any).mockResolvedValue({
      total: 1,
      limit: 20,
      offset: 0,
      items: [terminalAlert],
    });

    render(<AlertsPage />);

    await waitFor(() => {
      expect(screen.getByText('Inspect')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Inspect'));

    expect(
      screen.getByText(/This alert has been finalized in a terminal state/i)
    ).toBeInTheDocument();
    expect(screen.queryByText('Commit Triage Update')).not.toBeInTheDocument();
  });
});
