"use client";
import { useState, useEffect, useRef } from "react";
import Spinner from "./Spinner";
import type { HistoryPoint } from "@/hooks/useEnvironment";

interface PerformanceTrendProps {
  rewardHistory?: HistoryPoint[];
}

// ── SVG viewport constants ──────────────────────────────────────────────────
const VW = 100; // viewBox width
const VH = 50;  // viewBox height
const PAD_TOP = 3;
const PAD_BTM = 2;

/** Map a value in [dataMin, dataMax] to SVG Y (top = small Y) */
function toY(value: number, dataMin: number, dataMax: number): number {
  const range = dataMax - dataMin || 1;
  const normalised = (value - dataMin) / range; // 0 = min, 1 = max
  return PAD_TOP + (1 - normalised) * (VH - PAD_TOP - PAD_BTM);
}

/** Map step index to SVG X */
function toX(i: number, total: number): number {
  if (total <= 1) return VW / 2;
  return (i / (total - 1)) * VW;
}

/**
 * Build a smooth SVG path string using cardinal spline approximation.
 * Each segment is a cubic bezier with control points derived from neighbours.
 */
function buildSmoothPath(points: [number, number][]): string {
  if (points.length === 0) return "";
  if (points.length === 1) return `M${points[0][0]},${points[0][1]}`;

  const tension = 0.4;
  let d = `M${points[0][0]},${points[0][1]}`;

  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];

    const cp1x = p1[0] + (p2[0] - p0[0]) * tension;
    const cp1y = p1[1] + (p2[1] - p0[1]) * tension;
    const cp2x = p2[0] - (p3[0] - p1[0]) * tension;
    const cp2y = p2[1] - (p3[1] - p1[1]) * tension;

    d += ` C${cp1x},${cp1y} ${cp2x},${cp2y} ${p2[0]},${p2[1]}`;
  }
  return d;
}

/** Build a closed filled area path (line path + close to bottom) */
function buildAreaPath(points: [number, number][], linePath: string): string {
  if (points.length === 0) return "";
  const first = points[0];
  const last = points[points.length - 1];
  return `${linePath} L${last[0]},${VH} L${first[0]},${VH} Z`;
}

export default function PerformanceTrend({ rewardHistory = [] }: PerformanceTrendProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isResizing, setIsResizing] = useState(false);
  const resizeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Resize observer ────────────────────────────────────────────────────────
  // Only blank the chart when the container WIDTH genuinely changes (sidebar
  // toggle). Ignore height-only changes and tiny jitter from SVG re-renders
  // so the chart never flickers during normal 800ms polling.
  const prevWidthRef = useRef<number>(0);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    prevWidthRef.current = el.offsetWidth;

    const observer = new ResizeObserver((entries) => {
      const newWidth = entries[0]?.contentRect.width ?? el.offsetWidth;
      const delta = Math.abs(newWidth - prevWidthRef.current);
      if (delta < 3) return; // ignore sub-pixel jitter
      prevWidthRef.current = newWidth;

      setIsResizing(true);
      if (resizeTimer.current) clearTimeout(resizeTimer.current);
      resizeTimer.current = setTimeout(() => setIsResizing(false), 250);
    });
    observer.observe(el);
    return () => {
      observer.disconnect();
      if (resizeTimer.current) clearTimeout(resizeTimer.current);
    };
  }, []);

  const hasData = rewardHistory.length >= 1;
  const currentReward = rewardHistory[rewardHistory.length - 1]?.reward ?? 0;
  const prevReward = rewardHistory[rewardHistory.length - 2]?.reward ?? 0;
  const delta = currentReward - (rewardHistory[0]?.reward ?? 0);

  // ── Compute SVG paths ──────────────────────────────────────────────────────
  const { rewardLinePath, rewardAreaPath, healthLinePath, healthAreaPath, liveX, liveY } =
    (() => {
      if (!hasData) return { rewardLinePath: "", rewardAreaPath: "", healthLinePath: "", healthAreaPath: "", liveX: 0, liveY: 0 };

      const n = rewardHistory.length;

      // Reward: scale to actual reward range (can be negative)
      const rewardVals = rewardHistory.map((p) => p.reward);
      const rMin = Math.min(...rewardVals);
      const rMax = Math.max(...rewardVals, rMin + 0.01); // ensure range > 0

      // Health: always [0, 1]
      const hMin = 0;
      const hMax = 1;

      const rewardPts: [number, number][] = rewardHistory.map((p, i) => [
        toX(i, n),
        toY(p.reward, rMin, rMax),
      ]);
      const healthPts: [number, number][] = rewardHistory.map((p, i) => [
        toX(i, n),
        toY(p.health, hMin, hMax),
      ]);

      const rLine = buildSmoothPath(rewardPts);
      const hLine = buildSmoothPath(healthPts);

      const lastR = rewardPts[rewardPts.length - 1];

      return {
        rewardLinePath: rLine,
        rewardAreaPath: buildAreaPath(rewardPts, rLine),
        healthLinePath: hLine,
        healthAreaPath: buildAreaPath(healthPts, hLine),
        liveX: lastR[0],
        liveY: lastR[1],
      };
    })();

  // ── Y-axis tick labels for reward ──────────────────────────────────────────
  const rewardVals = rewardHistory.map((p) => p.reward);
  const rMin = Math.min(...rewardVals, 0);
  const rMax = Math.max(...rewardVals, 0.01);
  // Use index-keyed tuples to avoid duplicate key warnings when values round to same string
  const yTicks: [string, number][] = [
    [rMax.toFixed(2), 0],
    [((rMin + rMax) / 2).toFixed(2), 1],
    [rMin.toFixed(2), 2],
  ];

  // ── X-axis step labels (first, mid, last) ─────────────────────────────────
  const xLabels: { label: string; pct: string }[] = [];
  if (rewardHistory.length >= 1) {
    xLabels.push({ label: `S${rewardHistory[0].step}`, pct: "0%" });
  }
  if (rewardHistory.length >= 3) {
    const mid = Math.floor(rewardHistory.length / 2);
    xLabels.push({ label: `S${rewardHistory[mid].step}`, pct: "50%" });
  }
  if (rewardHistory.length >= 2) {
    xLabels.push({ label: `S${rewardHistory[rewardHistory.length - 1].step}`, pct: "100%" });
  }

  return (
    <>
      <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest">
        Performance Trend
      </h3>
      <div className="bg-[#1E293B]/60 backdrop-blur-xl border border-white/10 rounded-xl p-6" ref={containerRef}>
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-3 flex-wrap">
            <h4 className="font-h2 text-body-lg text-on-surface">System Reward Curve</h4>
            <span
              className={`font-mono text-sm px-2 py-0.5 rounded border transition-colors duration-500 ${delta >= 0
                  ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
                  : "text-red-400 bg-red-500/10 border-red-500/20"
                }`}
            >
              {delta >= 0 ? "+" : ""}
              {delta.toFixed(3)}
            </span>
            {/* Step count chip */}
            {hasData && (
              <span className="font-mono text-[11px] px-2 py-0.5 rounded border text-on-surface-variant bg-white/5 border-white/10">
                {rewardHistory.length} step{rewardHistory.length !== 1 ? "s" : ""}
              </span>
            )}
          </div>
          <span className="material-symbols-outlined text-outline flex-shrink-0">show_chart</span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-[#4D8EFF] rounded-full" />
            <span className="text-caption text-on-surface-variant">Reward</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 border-t border-dashed border-[#4ade80]" />
            <span className="text-caption text-on-surface-variant">Health</span>
          </div>
        </div>

        {/* Chart */}
        <div className="h-52 w-full relative flex items-end">
          {isResizing ? (
            <Spinner />
          ) : !hasData ? (
            <div className="flex flex-col items-center justify-center w-full h-full text-on-surface-variant/40 gap-2">
              <span className="material-symbols-outlined text-3xl">timeline</span>
              <p className="font-caption text-caption">No simulation data yet</p>
              <p className="font-caption text-[10px] text-on-surface-variant/30">
                POST /reset → run actions to see curves update
              </p>
            </div>
          ) : (
            <>
              {/* Y-axis labels */}
              <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-[10px] font-mono text-outline-variant py-1 pr-2 select-none">
                {yTicks.map(([label, idx]) => (
                  <span key={idx}>{label}</span>
                ))}
              </div>

              {/* Chart area */}
              <div className="ml-10 w-full h-full relative overflow-visible">
                {/* Grid lines */}
                <div className="absolute inset-0 flex flex-col justify-between py-2 pointer-events-none">
                  {[0, 1, 2, 3].map((i) => (
                    <div key={i} className="w-full h-[1px] bg-white/5" />
                  ))}
                </div>

                {/* SVG curves */}
                <svg
                  className="absolute w-full h-full overflow-visible"
                  preserveAspectRatio="none"
                  viewBox={`0 0 ${VW} ${VH}`}
                >
                  <defs>
                    <linearGradient id="ptIndigoGrad" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="0%" stopColor="#4D8EFF" stopOpacity="0.45" />
                      <stop offset="100%" stopColor="#4D8EFF" stopOpacity="0" />
                    </linearGradient>
                    <linearGradient id="ptGreenGrad" x1="0" x2="0" y1="0" y2="1">
                      <stop offset="0%" stopColor="#4ade80" stopOpacity="0.25" />
                      <stop offset="100%" stopColor="#4ade80" stopOpacity="0" />
                    </linearGradient>
                  </defs>

                  {/* Health area + line */}
                  {healthAreaPath && (
                    <path d={healthAreaPath} fill="url(#ptGreenGrad)" />
                  )}
                  {healthLinePath && (
                    <path
                      d={healthLinePath}
                      fill="none"
                      stroke="#4ade80"
                      strokeDasharray="2 2"
                      strokeWidth="1.2"
                      strokeLinecap="round"
                    />
                  )}

                  {/* Reward area + line */}
                  {rewardAreaPath && (
                    <path d={rewardAreaPath} fill="url(#ptIndigoGrad)" />
                  )}
                  {rewardLinePath && (
                    <path
                      d={rewardLinePath}
                      fill="none"
                      stroke="#4D8EFF"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  )}

                  {/* Data point dots on reward line */}
                  {rewardHistory.map((p, i) => {
                    const n = rewardHistory.length;
                    const rMin2 = Math.min(...rewardHistory.map((x) => x.reward));
                    const rMax2 = Math.max(...rewardHistory.map((x) => x.reward), rMin2 + 0.01);
                    const x = toX(i, n);
                    const y = toY(p.reward, rMin2, rMax2);
                    const isLast = i === n - 1;
                    return (
                      <circle
                        key={i}
                        cx={x}
                        cy={y}
                        r={isLast ? 2.2 : 1.2}
                        fill={isLast ? "#fff" : "#4D8EFF"}
                        stroke={isLast ? "#4D8EFF" : "none"}
                        strokeWidth={isLast ? 1.2 : 0}
                        style={{ transition: "cx 0.5s ease, cy 0.5s ease" }}
                      />
                    );
                  })}
                </svg>

                {/* Animated live endpoint glow dot */}
                <div
                  className="absolute w-3 h-3 bg-background rounded-full border-2 border-primary shadow-[0_0_10px_rgba(77,142,255,0.7)] transition-all duration-500 ease-out"
                  style={{
                    left: `calc(${(liveX / VW) * 100}% - 6px)`,
                    top: `calc(${(liveY / VH) * 100}% - 6px)`,
                  }}
                />
              </div>
            </>
          )}
        </div>

        {/* X-axis step labels */}
        {hasData && !isResizing && (
          <div className="ml-10 mt-1 relative h-4">
            {xLabels.map(({ label, pct }) => (
              <span
                key={pct}
                className="absolute text-[10px] font-mono text-outline-variant select-none -translate-x-1/2"
                style={{ left: pct }}
              >
                {label}
              </span>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
