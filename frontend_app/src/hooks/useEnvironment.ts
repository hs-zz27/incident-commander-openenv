import { useState, useEffect, useCallback, useRef } from 'react';

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, '') || '';

export type ServiceStatus = 'healthy' | 'degraded' | 'down';

export interface ServiceState {
  name: string;
  status: ServiceStatus;
  error_rate: number;
  latency_ms: number;
  cpu_percent: number;
  memory_percent: number;
  instances: number;
  version: string;
  log_quality: string;
}

export interface TimelineEvent {
  step: number;
  // Rich fields from the real API
  event: string;
  severity?: string;
  health?: number;
  health_delta?: number;
  reward?: number;
  event_type?: string;
  description?: string;
  affected_services?: string[];
  chaos_event?: string;
  // Legacy fields kept for compatibility
  actor?: string;
  details?: Record<string, any>;
  // Injected by the hook: marks brand-new entries for animation
  isNew?: boolean;
}

export interface RunbookEntry {
  fingerprint?: string;
  incident_type?: string;
  task_name?: string;
  root_cause?: string;
  root_cause_service?: string;
  fix_sequence?: string[];
  score?: number;
}

export interface IncidentState {
  episode_id: string | null;
  step_count: number;
  task_name: string;
  is_resolved: boolean;
  /** True when the episode is terminal (matches observation `done`). */
  done?: boolean;
  cumulative_reward: number;
  actions_taken: string[];
  services: Record<string, ServiceState>;
  incident_timeline: TimelineEvent[];
  runbook_memory?: RunbookEntry[];
  /** Total persisted entries in the runbook bank (may be > matched suggestions). */
  runbook_bank_count?: number;
  /** Backend chaos injection toggle for this episode */
  chaos_mode_active?: boolean;
  chaos_tuning?: {
    injection_probability?: number;
    min_step?: number;
    guarantee_by_step?: number;
  };
  escalation_tier?: number;
  services_at_risk?: string[];
  metadata?: Record<string, any>;
}

export interface HistoryPoint {
  step: number;
  reward: number;
  health: number;
}

export interface ScoreBreakdown {
  recovery: number;
  efficiency: number;
  diagnostics: number;
  ordering: number;
  memory: number;
}

export interface AutoPilotStep {
  action: Record<string, string>;
  routing: { used_model: boolean; reason: string };
  reward: number;
  done: boolean;
  system_health: number;
}

export interface ModelConfig {
  base_model: string;
  adapter_path: string;
  device: 'auto' | 'cpu' | 'cuda' | 'mps';
}

interface UseEnvironmentReturn {
  state: IncidentState | null;
  isConnected: boolean;
  error: string | null;
  rewardHistory: HistoryPoint[];
  resetEnvironment: (taskName?: string, chaosMode?: boolean) => Promise<void>;
  // Auto-pilot
  isAutoPilotRunning: boolean;
  startAutoPilot: (opts?: { ensureReset?: () => Promise<void>; modelConfig?: ModelConfig }) => void;
  stopAutoPilot: () => void;
  autoPilotSteps: AutoPilotStep[];
  // Live score
  liveScore: { score: number; breakdown: ScoreBreakdown } | null;
  fetchLiveScore: () => Promise<void>;
  // Model info
  modelInfo: any | null;
  fetchModelInfo: () => Promise<void>;
}

export function useEnvironment(pollingIntervalMs = 800): UseEnvironmentReturn {
  const [state, setState] = useState<IncidentState | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Accumulate per-step reward + health history, reset on new episode
  const historyRef = useRef<HistoryPoint[]>([]);
  const [rewardHistory, setRewardHistory] = useState<HistoryPoint[]>([]);

  // Track the count of known timeline events to detect new entries
  const prevTimelineLen = useRef<number>(0);
  // Track episode id to reset the "new" count when episode changes
  const prevEpisodeId = useRef<string | null>(null);
  const episodeDoneRef = useRef(false);

  // Auto-pilot state
  const [isAutoPilotRunning, setIsAutoPilotRunning] = useState(false);
  const autoPilotRef = useRef(false);
  const [autoPilotSteps, setAutoPilotSteps] = useState<AutoPilotStep[]>([]);

  // Live score state
  const [liveScore, setLiveScore] = useState<{ score: number; breakdown: ScoreBreakdown } | null>(null);

  // Model info (backend lazy-load status)
  const [modelInfo, setModelInfo] = useState<any | null>(null);

  const fetchState = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/state`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      const incoming: IncidentState = data.state;
      episodeDoneRef.current = Boolean(incoming.done);

      const episodeChanged = incoming.episode_id !== prevEpisodeId.current;
      if (episodeChanged) {
        prevTimelineLen.current = 0;
        prevEpisodeId.current = incoming.episode_id;
        historyRef.current = [];
      }

      // Tag newly arrived timeline entries with isNew = true
      const timeline = (incoming.incident_timeline ?? []).map((evt, idx) => ({
        ...evt,
        isNew: idx >= prevTimelineLen.current,
      }));

      prevTimelineLen.current = timeline.length;

      // Build reward+health history from timeline health snapshots + live cumulative reward
      const timelineHealthMap = new Map<number, number>();
      for (const evt of timeline) {
        if (evt.health != null) {
          timelineHealthMap.set(evt.step, evt.health);
        }
      }

      // Merge with existing history — add the current step if not already there
      const currentStep = incoming.step_count;
      const currentHealth = incoming.services
        ? (() => {
            const svcList = Object.values(incoming.services);
            if (!svcList.length) return 1;
            const healthyCount = svcList.filter((s: any) => s.status === 'healthy').length;
            return healthyCount / svcList.length;
          })()
        : (timelineHealthMap.get(currentStep) ?? 1);

      // Only append if this step is new
      const lastInHistory = historyRef.current[historyRef.current.length - 1];
      if (!lastInHistory || lastInHistory.step !== currentStep) {
        historyRef.current = [
          ...historyRef.current,
          { step: currentStep, reward: incoming.cumulative_reward, health: currentHealth },
        ];
        setRewardHistory([...historyRef.current]);
      }

      setState({ ...incoming, incident_timeline: timeline });
      setIsConnected(true);
      setError(null);
    } catch (e) {
      setIsConnected(false);
      setError(e instanceof Error ? e.message : 'Unknown error occurred');
    }
  }, []);

  const resetEnvironment = useCallback(async (taskName?: string | any, chaosMode?: boolean) => {
    // React's onClick passes an Event object, so we must check if it's actually a string
    const finalTaskName = typeof taskName === 'string' ? taskName : 'random_incident';
    const finalChaos = typeof chaosMode === 'boolean' ? chaosMode : false;

    // Stop auto-pilot on reset
    autoPilotRef.current = false;
    setIsAutoPilotRunning(false);
    setAutoPilotSteps([]);
    setLiveScore(null);

    try {
      const response = await fetch(`${API_BASE}/reset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ task_name: finalTaskName, chaos_mode: finalChaos })
      });
      if (!response.ok) {
        throw new Error(`Reset failed: ${response.status}`);
      }
      await fetchState();
    } catch (e) {
      console.error('Failed to reset environment:', e);
    }
  }, [fetchState]);

  const fetchLiveScore = useCallback(async () => {
    if (episodeDoneRef.current) {
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/score`);
      if (response.ok) {
        const data = await response.json();
        setLiveScore({ score: data.score, breakdown: data.breakdown });
      }
    } catch (e) {
      // Silently fail — score is optional
    }
  }, []);

  const fetchModelInfo = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/model/info`);
      if (response.ok) {
        const data = await response.json();
        setModelInfo(data);
      }
    } catch (e) {
      // optional
    }
  }, []);

  // Auto-pilot: call /predict_and_step in a loop
  const startAutoPilot = useCallback((opts?: { ensureReset?: () => Promise<void>; modelConfig?: ModelConfig }) => {
    autoPilotRef.current = true;
    setIsAutoPilotRunning(true);
    setAutoPilotSteps([]);

    const runStep = async () => {
      if (!autoPilotRef.current) return;

      try {
        // If caller asked to ensure a reset (episode initialized), do it once.
        if (opts?.ensureReset) {
          await opts.ensureReset();
          opts.ensureReset = undefined;
        }

        const response = await fetch(`${API_BASE}/predict_and_step`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(opts?.modelConfig ?? {}),
        });

        if (!response.ok) {
          const text = await response.text();
          setError(`Auto-Pilot failed (${response.status}): ${text}`);
          autoPilotRef.current = false;
          setIsAutoPilotRunning(false);
          return;
        }

        const data = await response.json();
        setAutoPilotSteps(prev => [...prev, {
          action: data.action_taken,
          routing: data.routing,
          reward: data.reward,
          done: data.done,
          system_health: data.system_health,
        }]);

        // Refresh state and score
        await fetchState();
        await fetchLiveScore();
        await fetchModelInfo();

        if (data.done || !autoPilotRef.current) {
          autoPilotRef.current = false;
          setIsAutoPilotRunning(false);
          return;
        }

        // Small delay between steps for visual effect
        setTimeout(runStep, 800);
      } catch (e) {
        console.error('Auto-pilot step failed:', e);
        setError(e instanceof Error ? e.message : 'Auto-Pilot failed');
        autoPilotRef.current = false;
        setIsAutoPilotRunning(false);
      }
    };

    runStep();
  }, [fetchState, fetchLiveScore, fetchModelInfo]);

  const stopAutoPilot = useCallback(() => {
    autoPilotRef.current = false;
    setIsAutoPilotRunning(false);
  }, []);

  useEffect(() => {
    // Initial fetch (state first so episodeDoneRef is current before /score)
    const tick = async () => {
      await fetchState();
      await fetchLiveScore();
    };
    void tick();

    const intervalId = setInterval(() => {
      void tick();
    }, pollingIntervalMs);

    return () => clearInterval(intervalId);
  }, [fetchState, fetchLiveScore, pollingIntervalMs]);

  return {
    state, isConnected, error, rewardHistory, resetEnvironment,
    isAutoPilotRunning, startAutoPilot, stopAutoPilot, autoPilotSteps,
    liveScore, fetchLiveScore,
    modelInfo, fetchModelInfo,
  };
}
