"use client";
import { useState, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import ServiceMap from "@/components/ServiceMap";
import ActivityLog from "@/components/ActivityLog";
import MetricsGauges from "@/components/MetricsGauges";
import PerformanceTrend from "@/components/PerformanceTrend";
import { useEnvironment } from "@/hooks/useEnvironment";

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  const { state, isConnected, error, rewardHistory, resetEnvironment } = useEnvironment(800);

  // Handle window resize logic for responsive sidebar
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setSidebarCollapsed(true);
      }
    };

    // Initial check
    handleResize();

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleToggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <div className="flex w-full h-full min-h-screen">
      <Sidebar collapsed={sidebarCollapsed} state={state} onReset={resetEnvironment} />

      <div
        className={`flex-1 flex flex-col min-h-screen main-content-transition ${sidebarCollapsed ? 'expanded-main' : 'ml-60'}`}
        id="main-wrapper"
      >
        <Header onToggleSidebar={handleToggleSidebar} isConnected={isConnected} taskName={state?.task_name} />

        <main className="flex-1 pt-8 px-8 pb-12 overflow-y-auto">
          <div className="mb-stack-lg max-w-[1400px] mx-auto w-full flex justify-between items-end">
            <div>
              <h2 className="font-h1 text-h1 text-on-surface">Service Map Overview</h2>
              <p className="font-body-md text-body-md text-on-surface-variant mt-1">Real-time status of interconnected core systems.</p>
            </div>
            {error && (
              <div className="bg-error/20 text-error px-4 py-2 rounded-lg border border-error/50 font-body-sm">
                Connection Error: {error}
              </div>
            )}
          </div>

          <div className="responsive-dashboard-grid max-w-[1400px] mx-auto w-full">
            <div className="grid-col-span-core flex flex-col gap-stack-md min-w-0">
              <ServiceMap services={state?.services || {}} />
            </div>

            <div className="grid-col-span-activity flex flex-col gap-stack-md min-w-0">
              <ActivityLog timeline={state?.incident_timeline || []} />
            </div>

            <div className="grid-col-span-metrics flex flex-col gap-stack-md min-w-0">
              <MetricsGauges services={state?.services || {}} />
            </div>

            <div className="grid-col-span-perf flex flex-col gap-stack-md min-w-0 mt-4">
              <PerformanceTrend rewardHistory={rewardHistory} />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
