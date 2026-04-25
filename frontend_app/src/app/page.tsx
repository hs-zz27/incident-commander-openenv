"use client";
import { useState, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import ServiceMap from "@/components/ServiceMap";
import ActivityLog from "@/components/ActivityLog";
import MetricsGauges from "@/components/MetricsGauges";
import PerformanceTrend from "@/components/PerformanceTrend";

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

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

  return (
    <div className="flex w-full h-full min-h-screen">
      <Sidebar collapsed={sidebarCollapsed} />

      <div
        className={`flex-1 flex flex-col min-h-screen main-content-transition ${sidebarCollapsed ? 'expanded-main' : 'ml-60'}`}
        id="main-wrapper"
      >
        <Header onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)} />

        <main className="flex-1 pt-8 px-8 pb-12 overflow-y-auto">
          <div className="mb-stack-lg max-w-[1400px] mx-auto w-full">
            <h2 className="font-h1 text-h1 text-on-surface">Service Map Overview</h2>
            <p className="font-body-md text-body-md text-on-surface-variant mt-1">Real-time status of interconnected core systems.</p>
          </div>

          <div className="responsive-dashboard-grid max-w-[1400px] mx-auto w-full">
            <div className="grid-col-span-core flex flex-col gap-stack-md min-w-0">
              <ServiceMap />
            </div>

            <div className="grid-col-span-activity flex flex-col gap-stack-md min-w-0">
              <ActivityLog />
            </div>

            <div className="grid-col-span-metrics flex flex-col gap-stack-md min-w-0">
              <MetricsGauges />
            </div>

            <div className="grid-col-span-perf flex flex-col gap-stack-md min-w-0 mt-4">
              <PerformanceTrend />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
