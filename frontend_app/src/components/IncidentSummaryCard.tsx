"use client";

import type { IncidentState, TimelineEvent } from "@/hooks/useEnvironment";

interface IncidentSummaryCardProps {
  state: IncidentState | null;
  isSimRunning?: boolean;
}

function titleCase(text: string): string {
  return text
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function extractServiceName(action: string): string | null {
  const parts = action.split(":");
  return parts.length > 1 ? parts[1] : null;
}

function humanizeAction(action: string): string {
  if (!action) return "Unknown action";
  const [actionType, serviceName] = action.split(":");

  const templates: Record<string, string> = {
    inspect_logs: "Checked logs",
    inspect_metrics: "Checked metrics",
    restart_service: "Restarted service",
    scale_service: "Scaled service",
    rollback: "Rolled back service",
    clear_cache: "Cleared cache",
    escalate: "Escalated incident",
    do_nothing: "No action taken",
    write_runbook: "Wrote runbook entry",
  };

  const label = templates[actionType] ?? titleCase(actionType);
  if (serviceName) return `${label}: ${serviceName}`;
  return label;
}

function buildSummary(state: IncidentState) {
  const actions = state.actions_taken ?? [];
  const timeline = state.incident_timeline ?? [];

  const inspectActions = actions.filter((a) => a.startsWith("inspect_"));
  const fixActions = actions.filter(
    (a) =>
      a.startsWith("restart_service:") ||
      a.startsWith("scale_service:") ||
      a.startsWith("rollback:") ||
      a === "clear_cache"
  );

  const serviceTouches = new Map<string, number>();
  for (const action of actions) {
    const service = extractServiceName(action);
    if (!service) continue;
    serviceTouches.set(service, (serviceTouches.get(service) ?? 0) + 1);
  }
  const rankedServices = [...serviceTouches.entries()].sort((a, b) => b[1] - a[1]);
  const likelyRoot = rankedServices[0]?.[0] ?? "unknown";

  const repeatedActions = actions.length - new Set(actions).size;
  const inefficiencyPct = actions.length
    ? Math.round((repeatedActions / actions.length) * 100)
    : 0;

  const firstStep = timeline[0]?.step ?? 0;
  const lastStep = timeline[timeline.length - 1]?.step ?? state.step_count ?? 0;
  const totalSteps = Math.max(0, lastStep - firstStep + (lastStep > 0 ? 1 : 0));

  const errors = timeline
    .filter((e: TimelineEvent & { error?: string }) => Boolean(e.error))
    .slice(-3)
    .map((e: TimelineEvent & { error?: string }) => e.error as string);

  const recoveryHighlights = fixActions.slice(-4);
  const investigationHighlights = inspectActions.slice(0, 3);

  return {
    likelyRoot,
    totalSteps,
    repeatedActions,
    inefficiencyPct,
    recoveryHighlights,
    investigationHighlights,
    errors,
  };
}

export default function IncidentSummaryCard({ state, isSimRunning }: IncidentSummaryCardProps) {
  if (!state) return null;

  const hasEpisodeData =
    (state.actions_taken?.length ?? 0) > 0 || (state.incident_timeline?.length ?? 0) > 0;
  if (!hasEpisodeData) return null;

  const summary = buildSummary(state);
  const resolved = state.is_resolved;
  
  const timeline = state.incident_timeline ?? [];
  const lastEvent = timeline[timeline.length - 1];
  const isDone = resolved || (lastEvent && (lastEvent.event_type === 'episode_ended' || lastEvent.event_type === 'incident_resolved'));
  const isStopped = !isSimRunning && !resolved && !isDone;

  return (
    <div className="bg-[#1E293B]/60 backdrop-blur-xl border border-white/10 rounded-xl p-6">
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest">
            Incident Wrap-up
          </h3>
          <p className="font-body-md text-body-md text-on-surface-variant mt-1">
            Quick recap of the incident and what moved recovery forward.
          </p>
        </div>
        <span
          className={`font-mono text-xs px-2 py-1 rounded border ${
            resolved
              ? "text-emerald-300 bg-emerald-500/10 border-emerald-500/30"
              : isDone
              ? "text-red-300 bg-red-500/10 border-red-500/30"
              : isStopped
              ? "text-on-surface-variant bg-white/5 border-white/10"
              : "text-amber-300 bg-amber-500/10 border-amber-500/30"
          }`}
        >
          {resolved ? "Resolved" : isDone ? "Failed" : isStopped ? "Stopped" : "In Progress"}
        </span>
      </div>

      <div className="rounded-lg border border-cyan-400/20 bg-cyan-500/[0.06] p-4 mb-4">
        <p className="text-sm text-on-surface leading-relaxed">
          {resolved
            ? `Incident resolved in ${summary.totalSteps} steps. The model most likely focused on ${summary.likelyRoot} as the primary issue, and recovery actions in later steps helped stabilize the system.`
            : `Incident is still active after ${summary.totalSteps} steps. Current signals suggest ${summary.likelyRoot} may be the main pressure point; more targeted recovery actions may be needed.`}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
        <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
          <p className="text-[11px] text-on-surface-variant font-mono uppercase tracking-wide">
            Most Impacted Service
          </p>
          <p className="mt-1 text-on-surface font-semibold">{titleCase(summary.likelyRoot)}</p>
        </div>
        <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
          <p className="text-[11px] text-on-surface-variant font-mono uppercase tracking-wide">
            Resolution Time
          </p>
          <p className="mt-1 text-on-surface font-semibold">{summary.totalSteps} steps</p>
        </div>
        <div className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
          <p className="text-[11px] text-on-surface-variant font-mono uppercase tracking-wide">
            Repeated Actions
          </p>
          <p className="mt-1 text-on-surface font-semibold">
            {summary.inefficiencyPct}% ({summary.repeatedActions} repeats)
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-lg border border-white/10 bg-[#0F172A]/60 p-4">
          <p className="text-[11px] text-cyan-300 font-mono uppercase tracking-wide mb-2">
            Actions That Helped
          </p>
          {summary.recoveryHighlights.length ? (
            <ul className="space-y-1 text-sm text-on-surface">
              {summary.recoveryHighlights.map((a, i) => (
                <li key={`${a}-${i}`}>- {humanizeAction(a)}</li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-on-surface-variant">
              No clear recovery action yet. Try restart, scale, rollback, or cache clear actions.
            </p>
          )}
        </div>

        <div className="rounded-lg border border-white/10 bg-[#0F172A]/60 p-4">
          <p className="text-[11px] text-emerald-300 font-mono uppercase tracking-wide mb-2">
            Investigation Steps
          </p>
          {summary.investigationHighlights.length ? (
            <ul className="space-y-1 text-sm text-on-surface">
              {summary.investigationHighlights.map((a, i) => (
                <li key={`${a}-${i}`}>- {humanizeAction(a)}</li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-on-surface-variant">
              No inspection actions recorded yet.
            </p>
          )}
        </div>
      </div>

      <div className="mt-4 rounded-lg border border-white/10 bg-[#0F172A]/60 p-4">
        <p className="text-[11px] text-amber-300 font-mono uppercase tracking-wide mb-2">
          Important Error Notes
        </p>
        {summary.errors.length ? (
          <ul className="space-y-1 text-sm text-on-surface">
            {summary.errors.map((err, i) => (
              <li key={`${err}-${i}`}>- {err}</li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-on-surface-variant">
            No explicit action errors were captured.
          </p>
        )}
      </div>
    </div>
  );
}
