"use client";

import type { RunbookEntry } from "@/hooks/useEnvironment";

interface RunbookModalProps {
  isOpen: boolean;
  onClose: () => void;
  entries: RunbookEntry[];
  bankCount?: number;
}

export default function RunbookModal({ isOpen, onClose, entries, bankCount }: RunbookModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[58] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-surface-container rounded-xl border border-cyan-500/25 shadow-2xl w-full max-w-lg max-h-[85vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between p-4 border-b border-outline-variant/10 bg-surface">
          <div>
            <h2 className="text-lg font-h2 font-bold text-on-surface">Runbook memory</h2>
            {bankCount != null && (
              <p className="text-[11px] text-on-surface-variant mt-0.5 font-mono">
                Bank: {bankCount} {bankCount === 1 ? "entry" : "entries"}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-md text-on-surface-variant hover:bg-outline-variant/20"
            aria-label="Close runbook"
          >
            <span className="material-symbols-outlined">close</span>
          </button>
        </div>
        <div className="p-4 overflow-y-auto flex-1 space-y-3">
          {entries.length === 0 ? (
            <p className="text-sm text-on-surface-variant">
              No fingerprint matches for this incident yet. After you resolve more episodes, similar
              incidents will surface suggestions here.
            </p>
          ) : (
            entries.map((entry, i) => (
              <div
                key={`${entry.incident_type ?? entry.fingerprint ?? i}-${i}`}
                className="rounded-lg border border-outline-variant/20 bg-surface-container-high/40 p-3 text-sm"
              >
                <p className="font-mono text-xs text-cyan-300/90 truncate" title={entry.incident_type}>
                  {entry.incident_type ?? entry.fingerprint ?? "suggestion"}
                </p>
                {(entry.root_cause_service || entry.root_cause) && (
                  <p className="text-xs text-on-surface-variant mt-1">
                    Root cause:{" "}
                    <span className="text-on-surface font-medium">
                      {entry.root_cause_service ?? entry.root_cause}
                    </span>
                  </p>
                )}
                {entry.fix_sequence && entry.fix_sequence.length > 0 && (
                  <p className="text-xs text-on-surface-variant mt-2 leading-relaxed">
                    <span className="text-on-surface-variant/70">Fix path:</span>{" "}
                    <span className="font-mono text-on-surface/90">{entry.fix_sequence.join(" → ")}</span>
                  </p>
                )}
              </div>
            ))
          )}
        </div>
        <div className="p-3 border-t border-outline-variant/10 flex justify-end bg-surface">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 bg-primary/20 text-primary border border-primary/30 rounded text-sm font-semibold hover:bg-primary/30"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
