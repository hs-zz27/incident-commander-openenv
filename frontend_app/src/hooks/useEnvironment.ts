import { useState, useEffect, useCallback, useRef } from 'react';

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
  description?: string;
  affected_services?: string[];
  // Legacy fields kept for compatibility
  actor?: string;
  details?: Record<string, any>;
  // Injected by the hook: marks brand-new entries for animation
  isNew?: boolean;
}

export interface IncidentState {
  episode_id: string | null;
  step_count: number;
  task_name: string;
  is_resolved: boolean;
  cumulative_reward: number;
  actions_taken: string[];
  services: Record<string, ServiceState>;
  incident_timeline: TimelineEvent[];
}

export interface HistoryPoint {
  step: number;
  reward: number;
  health: number;
}

interface UseEnvironmentReturn {
  state: IncidentState | null;
  isConnected: boolean;
  error: string | null;
  rewardHistory: HistoryPoint[];
  resetEnvironment: (taskName?: string) => Promise<void>;
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

  const fetchState = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/state');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      const incoming: IncidentState = data.state;

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
      // Each timeline event has a `health` field (0-1). We map step → health.
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

  const resetEnvironment = useCallback(async (taskName?: string | any) => {
    // React's onClick passes an Event object, so we must check if it's actually a string
    const finalTaskName = typeof taskName === 'string' ? taskName : 'random_incident';

    try {
      const response = await fetch('http://localhost:8000/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ task_name: finalTaskName })
      });
      if (!response.ok) {
        throw new Error(`Reset failed: ${response.status}`);
      }
      await fetchState();
    } catch (e) {
      console.error('Failed to reset environment:', e);
    }
  }, [fetchState]);

  useEffect(() => {
    // Initial fetch
    fetchState();

    // Polling
    const intervalId = setInterval(fetchState, pollingIntervalMs);

    return () => clearInterval(intervalId);
  }, [fetchState, pollingIntervalMs]);

  return { state, isConnected, error, rewardHistory, resetEnvironment };
}
