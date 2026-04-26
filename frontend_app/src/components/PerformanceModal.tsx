"use client";

import type { RunbookEntry, ScoreBreakdown } from "@/hooks/useEnvironment";

interface PerformanceModalProps {
  isOpen: boolean;
  onClose: () => void;
  taskName?: string;
  stepCount?: number;
  score: number;
  breakdown: ScoreBreakdown;
  lastChaosEvent?: string | null;
  runbookMemory?: RunbookEntry[];
  runbookBankCount?: number;
}

const MAX: Record<keyof ScoreBreakdown, number> = {
  recovery: 0.35,
  efficiency: 0.20,
  diagnostics: 0.15,
  ordering: 0.20,
  memory: 0.10,
};

export default function PerformanceModal({
  isOpen,
  onClose,
  taskName,
  stepCount,
  score,
  breakdown,
  lastChaosEvent,
  runbookMemory,
  runbookBankCount,
}: PerformanceModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-surface-container rounded-xl border border-primary/25 shadow-2xl w-full max-w-lg max-h-[90vh] flex flex-col overflow-hidden animate-in fade-in zoom-in duration-300">
        <div className="flex items-center justify-between p-4 border-b border-outline-variant/10 bg-surface">
          <div>
            <h2 className="text-lg font-h2 font-bold text-on-surface">Incident complete</h2>
            <p className="text-[11px] text-on-surface-variant mt-0.5 font-mono uppercase tracking-wide">
              {taskName ? taskName.replace(/_/g, " ") : "Episode"}
              {stepCount != null ? ` · ${stepCount} steps` : ""}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-md text-on-surface-variant hover:bg-outline-variant/20 hover:text-on-surface transition-colors"
            aria-label="Close performance summary"
          >
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>

        <div className="p-6 flex flex-col gap-6">
          <div className="flex flex-col items-center border-b border-outline-variant/10 pb-4">
            <span className="text-xs text-on-surface-variant/70 uppercase tracking-widest font-semibold mb-1">
              Final score
            </span>
            <div className="text-5xl font-display font-bold text-primary drop-shadow-md">
              {score.toFixed(3)}
            </div>
            <p className="text-[11px] text-on-surface-variant mt-2 text-center">
              Score is frozen at episode end so it won&apos;t drift while the UI polls the server.
            </p>
          </div>

          <div className="space-y-3">
            {(Object.keys(breakdown) as (keyof ScoreBreakdown)[]).map((key) => {
              const val = breakdown[key] ?? 0;
              const max = MAX[key] ?? 0.2;
              const pct = max > 0 ? Math.min(100, (val / max) * 100) : 0;
              return (
                <div key={key} className="flex items-center gap-2">
                  <span className="text-xs text-on-surface-variant/80 capitalize w-28 shrink-0">
                    {key}
                  </span>
                  <div className="flex-1 h-2 bg-outline-variant/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        pct >= 90 ? "bg-green-500" : pct >= 60 ? "bg-yellow-500" : "bg-red-500"
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs text-on-surface-variant/60 w-14 text-right font-mono">
                    {val.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>

          {(lastChaosEvent ||
            (runbookMemory && runbookMemory.length > 0) ||
            runbookBankCount != null) && (
            <div className="rounded-lg border border-outline-variant/20 bg-surface-container-high/30 p-4 space-y-3">
              <p className="text-[10px] uppercase tracking-widest text-on-surface-variant font-semibold">
                Episode context
              </p>
              {lastChaosEvent ? (
                <div className="text-sm">
                  <span className="text-on-surface-variant text-xs uppercase tracking-wide">
                    Last chaos target
                  </span>
                  <p className="font-mono text-fuchsia-200 mt-0.5">{lastChaosEvent}</p>
                </div>
              ) : null}
              {runbookBankCount != null ? (
                <p className="text-xs text-on-surface-variant">
                  Runbook bank:{" "}
                  <span className="font-mono text-on-surface">{runbookBankCount}</span>{" "}
                  {runbookBankCount === 1 ? "entry" : "entries"}
                </p>
              ) : null}
              {runbookMemory && runbookMemory.length > 0 ? (
                <div>
                  <p className="text-xs text-on-surface-variant mb-2">Matching suggestions</p>
                  <ul className="space-y-2 max-h-36 overflow-y-auto text-xs">
                    {runbookMemory.map((entry, i) => (
                      <li
                        key={`${entry.incident_type ?? i}-${i}`}
                        className="rounded border border-outline-variant/15 bg-black/20 p-2 font-mono text-[11px] text-on-surface-variant"
                      >
                        <div className="truncate text-cyan-300/90">
                          {entry.incident_type ?? entry.fingerprint ?? "match"}
                        </div>
                        {entry.root_cause_service ? (
                          <div className="mt-1 normal-case text-on-surface">
                            RC: {entry.root_cause_service}
                          </div>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="text-xs text-on-surface-variant/80">
                  No runbook fingerprint matches for this incident type.
                </p>
              )}
            </div>
          )}
        </div>

        <div className="p-4 border-t border-outline-variant/10 bg-surface flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 bg-primary/20 text-primary border border-primary/30 rounded font-semibold text-sm hover:bg-primary/30 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
