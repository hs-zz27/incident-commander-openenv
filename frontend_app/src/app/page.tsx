"use client";
import { useState, useEffect, useCallback, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import ServiceMap from "@/components/ServiceMap";
import ActivityLog from "@/components/ActivityLog";
import PerformanceTrend from "@/components/PerformanceTrend";
import IncidentSummaryCard from "@/components/IncidentSummaryCard";
import PerformanceModal from "@/components/PerformanceModal";
import ChaosToast from "@/components/ChaosToast";
import InsightDock from "@/components/InsightDock";
import RunbookModal from "@/components/RunbookModal";
import { useEnvironment, type ScoreBreakdown, type RunbookEntry } from "@/hooks/useEnvironment";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, '') || '';

export default function Home() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [selectedTask, setSelectedTask] = useState('random_incident');
  const [chaosMode, setChaosMode] = useState(true);
  const [isSimRunning, setIsSimRunning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isWaitingForReset, setIsWaitingForReset] = useState(false);
  const [lastEpisodeId, setLastEpisodeId] = useState<string | null>(null);

  const {
    state,
    isConnected,
    error,
    rewardHistory,
    liveScore,
    resetEnvironment,
  } = useEnvironment(800);

  const [showPerformanceModal, setShowPerformanceModal] = useState(false);
  const [modalScore, setModalScore] = useState<{
    score: number;
    breakdown: ScoreBreakdown;
  } | null>(null);
  const [modalContext, setModalContext] = useState<{
    lastChaos?: string | null;
    runbookMemory: RunbookEntry[];
    runbookBankCount?: number;
  } | null>(null);
  const performanceModalOpenedForId = useRef<string | null>(null);
  const lastChaosToastKey = useRef<string | null>(null);
  const [chaosToastMessage, setChaosToastMessage] = useState<string | null>(null);
  const [showRunbookModal, setShowRunbookModal] = useState(false);

  useEffect(() => {
    performanceModalOpenedForId.current = null;
    lastChaosToastKey.current = null;
    setShowPerformanceModal(false);
    setModalScore(null);
    setModalContext(null);
    setChaosToastMessage(null);
    setShowRunbookModal(false);
  }, [state?.episode_id]);

  useEffect(() => {
    const done = Boolean(state?.done);
    const episodeId = state?.episode_id;
    if (!done || !episodeId) return;
    if (performanceModalOpenedForId.current === episodeId) return;

    const snap = state;
    void (async () => {
      try {
        const res = await fetch(`${API_BASE}/score`);
        if (!res.ok) return;
        const data = await res.json();
        performanceModalOpenedForId.current = episodeId;
        setModalScore({
          score: data.score as number,
          breakdown: data.breakdown as ScoreBreakdown,
        });
        setModalContext({
          lastChaos:
            (snap?.metadata?.last_chaos_event as string | undefined) ||
            (snap?.metadata?.new_chaos_event as string | undefined) ||
            null,
          runbookMemory: snap?.runbook_memory ?? [],
          runbookBankCount: snap?.runbook_bank_count,
        });
        setShowPerformanceModal(true);
      } catch {
        /* ignore */
      }
    })();
  }, [state?.done, state?.episode_id]);

  useEffect(() => {
    const neu = state?.metadata?.new_chaos_event as string | undefined;
    if (!neu || !state?.episode_id) return;
    const key = `${state.episode_id}:${state.step_count}:${neu}`;
    if (lastChaosToastKey.current === key) return;
    lastChaosToastKey.current = key;
    setChaosToastMessage(neu);
  }, [state?.metadata?.new_chaos_event, state?.episode_id, state?.step_count]);

  // Handle window resize logic for responsive sidebar
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setSidebarCollapsed(true);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Poll sim-status to track if live_inference.py is still running
  useEffect(() => {
    if (!isSimRunning) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/sim-status`);
        const data = await res.json();
        if (!data.running) {
          setIsSimRunning(false);
        }
      } catch {
        // Backend might be busy — ignore
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [isSimRunning]);

  // Clear the "waiting for reset" state once a new episode ID is observed
  useEffect(() => {
    if (isWaitingForReset && state?.episode_id && state.episode_id !== lastEpisodeId) {
      setIsWaitingForReset(false);
    }
  }, [state?.episode_id, lastEpisodeId, isWaitingForReset]);

  const handleToggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  const handleStartSim = useCallback(async () => {
    setIsStarting(true);
    setLastEpisodeId(state?.episode_id || null);
    setIsWaitingForReset(true);
    const body = JSON.stringify({ task: selectedTask, chaos: chaosMode });
    try {
      let res = await fetch(`${API_BASE}/start-sim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      if (res.status === 404) {
        res = await fetch(`${API_BASE}/start_sim`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
        });
      }
      if (res.ok) {
        setIsSimRunning(true);
      } else if (res.status === 404) {
        // Older or proxied servers may omit /start-sim; still start an episode.
        await resetEnvironment(selectedTask, chaosMode);
        console.warn(
          'POST /start-sim not found; started episode via /reset only (no live_inference subprocess).',
        );
      } else {
        const t = await res.text().catch(() => '');
        console.error('start-sim failed:', res.status, t);
      }
    } catch (err) {
      console.error('Failed to start sim:', err);
    } finally {
      setIsStarting(false);
      setIsWaitingForReset(false);
    }
  }, [selectedTask, chaosMode, state?.episode_id, resetEnvironment]);

  const handleStopSim = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/stop-sim`, { method: 'POST' });
    } catch {
      // ignore
    }
    setIsSimRunning(false);
  }, []);

  return (
    <div className="flex w-full h-full min-h-screen">
      <ChaosToast message={chaosToastMessage} />

      <InsightDock
        connected={isConnected}
        chaosModeActive={state?.chaos_mode_active}
        chaosTuning={state?.chaos_tuning}
        lastChaosEvent={state?.metadata?.last_chaos_event as string | undefined}
        newChaosEvent={state?.metadata?.new_chaos_event as string | undefined}
        runbookMatchCount={state?.runbook_memory?.length ?? 0}
        runbookBankCount={state?.runbook_bank_count}
        onOpenRunbook={() => setShowRunbookModal(true)}
      />

      <RunbookModal
        isOpen={showRunbookModal}
        onClose={() => setShowRunbookModal(false)}
        entries={state?.runbook_memory ?? []}
        bankCount={state?.runbook_bank_count}
      />

      <Sidebar
        collapsed={sidebarCollapsed}
        state={state}
        services={state?.services || {}}
      />

      <div
        className={`flex-1 flex flex-col min-h-screen main-content-transition ${sidebarCollapsed ? 'expanded-main' : 'ml-48'}`}
        id="main-wrapper"
      >
        <Header
          onToggleSidebar={handleToggleSidebar}
          isConnected={isConnected}
          taskName={state?.task_name}
          selectedTask={selectedTask}
          onSelectedTaskChange={setSelectedTask}
          chaosMode={chaosMode}
          onChaosModeChange={setChaosMode}
          isSimRunning={isSimRunning}
          isStarting={isStarting}
          onStartSim={handleStartSim}
          onStopSim={handleStopSim}
        />

        <main className="flex-1 pt-8 px-8 pb-12 overflow-y-auto scroll-smooth">
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
            <div id="service-map" className="grid-col-span-core flex flex-col gap-stack-md min-w-0">
              <ServiceMap services={state?.services || {}} isInitializing={isWaitingForReset} />
            </div>

            <div id="activity-log" className="grid-col-span-activity flex flex-col gap-stack-md min-w-0">
              <ActivityLog
                timeline={state?.incident_timeline || []}
                isSimRunning={isSimRunning}
                forceSpinner={isWaitingForReset}
              />
            </div>

            {/* Live Score */}
            <div id="live-score" className="grid-col-span-metrics flex flex-col gap-stack-md min-w-0">
              <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase">Live Score</h3>
              {liveScore && !isWaitingForReset ? (
                <div className="bg-surface-container-high/40 backdrop-blur-xl border border-primary/20 rounded-xl p-6 flex flex-col items-center">
                  <div className="flex flex-col items-center mb-6 w-full border-b border-outline-variant/10 pb-4">
                    <span className="text-xs text-on-surface-variant/70 uppercase tracking-widest font-semibold mb-1">Total Score</span>
                    <div className="text-6xl font-display font-bold text-primary drop-shadow-md">
                      {liveScore.score.toFixed(3)}
                    </div>
                  </div>
                  <div className="w-full space-y-2">
                    {Object.entries(liveScore.breakdown).filter(([key]) => key !== 'memory').map(([key, val]) => {
                      const maxVals: Record<string, number> = { recovery: 0.35, efficiency: 0.20, diagnostics: 0.15, ordering: 0.20 };
                      const max = maxVals[key] || 0.2;
                      const pct = max > 0 ? ((val as number) / max) * 100 : 0;
                      return (
                        <div key={key} className="flex items-center gap-2">
                          <div className="flex items-center gap-1 w-24 group relative cursor-help">
                            <span className="text-xs text-on-surface-variant/70 capitalize">{key}</span>
                            <span className="material-symbols-outlined text-xs text-on-surface-variant/40 group-hover:text-primary transition-colors">info</span>
                            {/* Tooltip */}
                            <div className="absolute left-0 bottom-full mb-1 hidden group-hover:block w-40 p-2 bg-surface-container-highest border border-outline-variant/20 rounded shadow-lg text-[9px] text-on-surface-variant z-10 font-normal normal-case leading-tight">
                              {key === 'recovery' && "Speed of restoring healthy service status."}
                              {key === 'efficiency' && "Minimizing unnecessary commands & actions."}
                              {key === 'diagnostics' && "Inspecting logs & metrics before acting."}
                              {key === 'ordering' && "Correct sequence of resolution steps."}
                            </div>
                          </div>
                          <div className="flex-1 h-2 bg-outline-variant/10 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${pct >= 90 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
                                }`}
                              style={{ width: `${Math.min(pct, 100)}%` }}
                            />
                          </div>
                          <span className="text-xs text-on-surface-variant/50 w-10 text-right font-mono">{(val as number).toFixed(2)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="bg-surface-container-high/40 backdrop-blur-xl border border-outline-variant/10 rounded-xl p-6 flex flex-col items-center justify-center text-on-surface-variant/40 gap-2 min-h-[120px]">
                  {isSimRunning || isWaitingForReset ? (
                    <>
                      <div className="w-6 h-6 border-2 border-primary/20 border-t-primary rounded-full animate-spin" />
                      <p className="text-[11px]">Calculating score…</p>
                    </>
                  ) : (
                    <>
                      <span className="material-symbols-outlined text-3xl opacity-40">leaderboard</span>
                      <p className="text-[11px]">Score appears during simulation</p>
                    </>
                  )}
                </div>
              )}
            </div>

            <div id="performance" className="grid-col-span-perf flex flex-col gap-stack-md min-w-0 mt-4">
              <PerformanceTrend rewardHistory={rewardHistory} />
              <IncidentSummaryCard state={state} isSimRunning={isSimRunning} />
            </div>
          </div>
        </main>
      </div>

      {modalScore && modalContext && (
        <PerformanceModal
          isOpen={showPerformanceModal}
          onClose={() => setShowPerformanceModal(false)}
          taskName={state?.task_name}
          stepCount={state?.step_count}
          score={modalScore.score}
          breakdown={modalScore.breakdown}
          lastChaosEvent={modalContext.lastChaos}
          runbookMemory={modalContext.runbookMemory}
          runbookBankCount={modalContext.runbookBankCount}
        />
      )}
    </div>
  );
}
