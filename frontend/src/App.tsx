import React, { useState, useEffect, useCallback } from 'react';
import { Sidebar, NavTab } from './components/Sidebar';
import { Topbar } from './components/Topbar';
import { OverviewPage } from './pages/OverviewPage';
import { AlertsPage } from './pages/AlertsPage';
import { PredictionPage } from './pages/PredictionPage';
import { ReplayPage } from './pages/ReplayPage';
import { StreamingPage } from './pages/StreamingPage';
import { api } from './api/client';
import { ServiceHealthStatus } from './types';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');
  const [systemStatus, setSystemStatus] = useState<ServiceHealthStatus | 'CONNECTING' | 'ERROR'>('CONNECTING');
  const [openAlertsCount, setOpenAlertsCount] = useState<number>(0);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [refreshTrigger, setRefreshTrigger] = useState<number>(0);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);

  const checkSystemHealth = useCallback(async () => {
    try {
      const health = await api.getHealth();
      setSystemStatus(health.status);
    } catch {
      setSystemStatus('ERROR');
    }
  }, []);

  const checkOpenAlerts = useCallback(async () => {
    try {
      const res = await api.getAlerts({ status: 'OPEN', limit: 1 });
      setOpenAlertsCount(res.total);
    } catch {
      // Ignored if backend is offline
    }
  }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    setRefreshTrigger((prev) => prev + 1);
    await Promise.allSettled([checkSystemHealth(), checkOpenAlerts()]);
    setLastUpdated(new Date());
    setIsRefreshing(false);
  };

  useEffect(() => {
    handleRefresh();

    // Light background poll every 30 seconds for health
    const interval = setInterval(() => {
      checkSystemHealth();
      checkOpenAlerts();
    }, 30000);

    return () => clearInterval(interval);
  }, [checkSystemHealth, checkOpenAlerts]);

  return (
    <div className="app-container">
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        systemStatus={systemStatus}
        openAlertsCount={openAlertsCount}
        isOpenMobile={isMobileMenuOpen}
        onCloseMobile={() => setIsMobileMenuOpen(false)}
      />

      <div className="main-content">
        <Topbar
          currentTab={currentTab}
          onRefresh={handleRefresh}
          isLoading={isRefreshing}
          lastUpdated={lastUpdated}
          onToggleMobileMenu={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        />

        <main className="content-area">
          {currentTab === 'overview' && (
            <OverviewPage
              onNavigate={setCurrentTab}
              onRefreshTrigger={refreshTrigger}
            />
          )}
          {currentTab === 'alerts' && (
            <AlertsPage onRefreshTrigger={refreshTrigger} />
          )}
          {currentTab === 'prediction' && <PredictionPage />}
          {currentTab === 'replay' && (
            <ReplayPage onRefreshTrigger={refreshTrigger} />
          )}
          {currentTab === 'streaming' && <StreamingPage />}
        </main>
      </div>
    </div>
  );
};

export default App;
