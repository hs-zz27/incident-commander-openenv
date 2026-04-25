'use client';
import { useEffect, useRef } from 'react';
import type { TimelineEvent } from '@/hooks/useEnvironment';

// ── Severity config ────────────────────────────────────────────────────────────
const SEVERITY_CONFIG: Record<string, { dot: string; bar: string; badge: string; icon: string }> = {
  critical: {
    dot: 'bg-red-400',
    bar: 'bg-red-500/20',
    badge: 'text-red-300 bg-red-500/15 border border-red-500/30',
    icon: 'emergency',
  },
  high: {
    dot: 'bg-orange-400',
    bar: 'bg-orange-500/20',
    badge: 'text-orange-300 bg-orange-500/15 border border-orange-500/30',
    icon: 'warning',
  },
  medium: {
    dot: 'bg-yellow-400',
    bar: 'bg-yellow-500/20',
    badge: 'text-yellow-300 bg-yellow-500/15 border border-yellow-500/30',
    icon: 'info',
  },
  low: {
    dot: 'bg-blue-400',
    bar: 'bg-blue-500/20',
    badge: 'text-blue-300 bg-blue-500/15 border border-blue-500/30',
    icon: 'notifications',
  },
  resolved: {
    dot: 'bg-emerald-400',
    bar: 'bg-emerald-500/20',
    badge: 'text-emerald-300 bg-emerald-500/15 border border-emerald-500/30',
    icon: 'check_circle',
  },
};

// ── Event-name → human label ──────────────────────────────────────────────────
function humanizeEvent(event: string): string {
  return event
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ── Actor config (legacy fallback) ────────────────────────────────────────────
function actorConfig(actor?: string) {
  if (actor === 'system' || actor === 'chaos')
    return { dot: 'bg-red-400', badge: 'text-red-300 bg-red-500/15 border border-red-500/30', label: 'System' };
  if (actor === 'agent')
    return { dot: 'bg-blue-400', badge: 'text-blue-300 bg-blue-500/15 border border-blue-500/30', label: 'Agent' };
  if (actor === 'environment')
    return { dot: 'bg-teal-400', badge: 'text-teal-300 bg-teal-500/15 border border-teal-500/30', label: 'Env' };
  return { dot: 'bg-slate-400', badge: 'text-slate-300 bg-slate-500/15 border border-slate-500/30', label: actor ?? 'Unknown' };
}

interface ActivityLogProps {
  timeline: TimelineEvent[];
}

export default function ActivityLog({ timeline }: ActivityLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to top whenever a new entry arrives (we render newest-first)
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [timeline.length]);

  // Show most recent events first, cap at 30
  const events = [...timeline].reverse().slice(0, 30);

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest">
          Activity Log
        </h3>
        <div className="flex items-center gap-2">
          {events.length > 0 && (
            <span className="text-[10px] font-semibold text-on-surface-variant/50 tabular-nums">
              {events.length} event{events.length !== 1 ? 's' : ''}
            </span>
          )}
          {/* Live pulse dot */}
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
        </div>
      </div>

      {/* Card */}
      <div className="bg-[#1E293B]/60 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden flex-1 flex flex-col">
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-5 space-y-3">
          {events.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-on-surface-variant/50 gap-3 py-12">
              <span className="material-symbols-outlined text-4xl opacity-40">hourglass_empty</span>
              <p className="font-caption text-caption">Waiting for simulation…</p>
              <p className="font-caption text-[10px] text-on-surface-variant/30">
                POST /reset → run inference.py
              </p>
            </div>
          ) : (
            events.map((evt, i) => {
              // Prefer severity-based styling, fall back to actor-based
              const sev = evt.severity ? SEVERITY_CONFIG[evt.severity] ?? SEVERITY_CONFIG.medium : null;
              const actor = actorConfig(evt.actor);
              const dotClass = sev ? sev.dot : actor.dot;
              const badgeClass = sev
                ? sev.badge
                : actor.badge;
              const badgeLabel = sev
                ? evt.severity!.charAt(0).toUpperCase() + evt.severity!.slice(1)
                : actor.label;
              const iconName = sev ? sev.icon : 'radio_button_checked';

              const healthPct = evt.health != null ? Math.round(evt.health * 100) : null;

              return (
                <div
                  key={`${evt.step}-${i}`}
                  // New entries slide in + fade from the top
                  className={`relative rounded-lg border border-white/5 overflow-hidden transition-all duration-500
                    ${evt.isNew ? 'activity-log-enter' : ''}
                  `}
                  style={
                    evt.isNew
                      ? { '--delay': `${i * 40}ms` } as React.CSSProperties
                      : {}
                  }
                >
                  {/* Left severity bar */}
                  <div className={`absolute inset-y-0 left-0 w-1 ${dotClass} rounded-l-lg`} />

                  {/* Content */}
                  <div className="pl-4 pr-3 py-3">
                    {/* Row 1: icon + event name + step badge */}
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className={`material-symbols-outlined text-[16px] flex-shrink-0 ${dotClass.replace('bg-', 'text-')}`}>
                          {iconName}
                        </span>
                        <span className="font-body-md text-body-md text-on-surface font-semibold truncate leading-tight">
                          {humanizeEvent(evt.event)}
                        </span>
                      </div>
                      <span className="flex-shrink-0 text-[10px] font-mono text-on-surface-variant/60 bg-white/5 px-2 py-0.5 rounded">
                        Step {evt.step}
                      </span>
                    </div>

                    {/* Row 2: description */}
                    {evt.description && (
                      <p className="mt-1.5 text-[12px] text-on-surface-variant leading-relaxed line-clamp-2">
                        {evt.description}
                      </p>
                    )}

                    {/* Row 3: badges row */}
                    <div className="flex flex-wrap items-center gap-1.5 mt-2">
                      {/* Severity / Actor badge */}
                      <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${badgeClass}`}>
                        {badgeLabel}
                      </span>

                      {/* Health score chip */}
                      {healthPct != null && (
                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border
                          ${healthPct >= 80
                            ? 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30'
                            : healthPct >= 50
                            ? 'text-yellow-300 bg-yellow-500/10 border-yellow-500/30'
                            : 'text-red-300 bg-red-500/10 border-red-500/30'
                          }`}
                        >
                          ♥ {healthPct}%
                        </span>
                      )}

                      {/* Affected services */}
                      {evt.affected_services?.map((svc) => (
                        <span
                          key={svc}
                          className="text-[10px] font-mono px-2 py-0.5 rounded-full text-primary/80 bg-primary/10 border border-primary/20"
                        >
                          {svc}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
