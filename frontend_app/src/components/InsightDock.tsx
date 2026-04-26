"use client";

interface ChaosTuning {
  injection_probability?: number;
  min_step?: number;
  guarantee_by_step?: number;
}

/**
 * Fixed dock — does not change the main dashboard grid layout.
 * Chaos ON/OFF reflects the *server episode* (chaos_mode_active), which is set
 * by POST /reset — e.g. when you press Start Sim with Chaos selected.
 */
interface InsightDockProps {
  connected: boolean;
  chaosModeActive?: boolean;
  chaosTuning?: ChaosTuning | null;
  lastChaosEvent?: string | null;
  newChaosEvent?: string | null;
  runbookMatchCount: number;
  runbookBankCount?: number;
  onOpenRunbook: () => void;
}

export default function InsightDock({
  connected,
  chaosModeActive,
  chaosTuning,
  lastChaosEvent,
  newChaosEvent,
  runbookMatchCount,
  runbookBankCount,
  onOpenRunbook,
}: InsightDockProps) {
  if (!connected) return null;

  const injecting = Boolean(chaosModeActive);
  const chaosLabel = newChaosEvent || lastChaosEvent;
  const chaosPulse = Boolean(newChaosEvent);

  const tune =
    chaosTuning &&
    chaosTuning.injection_probability != null &&
    chaosTuning.min_step != null &&
    chaosTuning.guarantee_by_step != null
      ? `p=${chaosTuning.injection_probability} · min step ${chaosTuning.min_step} · guarantee @${chaosTuning.guarantee_by_step}`
      : null;

  let chaosDetail = "—";
  if (injecting) {
    chaosDetail = chaosLabel ? `Target: ${chaosLabel}` : "No injection yet (random)";
  } else {
    chaosDetail = "Chaos off — use Start Sim with Chaos on";
  }

  return (
    <div className="fixed bottom-6 right-6 z-[52] flex flex-col gap-2 w-[min(100vw-2rem,17rem)] pointer-events-auto">
      <div
        className={`rounded-xl border px-3 py-2.5 backdrop-blur-md shadow-lg ${
          injecting
            ? "border-fuchsia-500/35 bg-fuchsia-950/40 text-fuchsia-100"
            : "border-outline-variant/30 bg-surface-container-highest/90 text-on-surface-variant"
        }`}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="text-[9px] uppercase tracking-widest font-bold text-on-surface-variant/90">
            Chaos
          </span>
          <span
            className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
              injecting ? "bg-fuchsia-500/25 text-fuchsia-100" : "bg-black/20 text-on-surface-variant"
            }`}
          >
            {injecting ? "ON" : "OFF"}
          </span>
        </div>
        {tune && (
          <p className="mt-1 text-[9px] text-on-surface-variant/80 font-mono normal-case tracking-normal">
            {tune}
          </p>
        )}
        <p
          className={`mt-1.5 text-xs font-mono break-words ${
            chaosPulse ? "text-fuchsia-50 animate-pulse" : "text-on-surface/90"
          }`}
        >
          {chaosDetail}
        </p>
      </div>

      <div className="rounded-xl border border-cyan-500/30 bg-slate-900/85 backdrop-blur-md px-3 py-2.5 shadow-lg text-on-surface">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[9px] uppercase tracking-widest font-bold text-cyan-200/90">Runbook</span>
          <button
            type="button"
            onClick={onOpenRunbook}
            className="text-[10px] font-semibold text-cyan-300 hover:text-cyan-100 underline-offset-2 hover:underline"
          >
            Details
          </button>
        </div>
        <p className="mt-1.5 text-[11px] text-on-surface-variant leading-snug">
          <span className="font-mono text-on-surface">{runbookMatchCount}</span> match
          {runbookMatchCount === 1 ? "" : "es"}
          {runbookBankCount != null && (
            <>
              {" · "}
              bank <span className="font-mono text-on-surface">{runbookBankCount}</span>
            </>
          )}
        </p>
      </div>
    </div>
  );
}
