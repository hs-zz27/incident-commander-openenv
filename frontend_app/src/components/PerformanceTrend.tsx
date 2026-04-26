"use client";

import { useMemo } from "react";
import type { HistoryPoint } from "@/hooks/useEnvironment";
import {
  Area,
  Brush,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface PerformanceTrendProps {
  rewardHistory?: HistoryPoint[];
}

interface ChartPoint {
  step: number;
  reward: number;
  healthPct: number;
}

interface TooltipEntry {
  dataKey?: string | number;
  name?: string | number;
  value?: number | string;
}

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: number | string;
}) {
  if (!active || !payload?.length) return null;

  const byKey = new Map<string, number>();
  for (const item of payload) {
    const key = String(item.dataKey ?? item.name ?? "");
    const value = Number(item.value ?? 0);
    if (!byKey.has(key)) byKey.set(key, value);
  }

  const reward = byKey.get("reward");
  const health = byKey.get("healthPct");

  return (
    <div
      style={{
        background: "rgba(15,23,42,0.92)",
        border: "1px solid rgba(148,163,184,0.25)",
        borderRadius: "10px",
        color: "#E2E8F0",
        fontSize: "12px",
        padding: "8px 10px",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: "6px" }}>Step {label}</div>
      {reward != null && <div>Reward: {reward.toFixed(3)}</div>}
      {health != null && <div>Health %: {health.toFixed(1)}%</div>}
    </div>
  );
}

export default function PerformanceTrend({ rewardHistory = [] }: PerformanceTrendProps) {
  const hasData = rewardHistory.length > 0;
  const currentReward = rewardHistory[rewardHistory.length - 1]?.reward ?? 0;
  const delta = currentReward - (rewardHistory[0]?.reward ?? 0);

  const data = useMemo<ChartPoint[]>(
    () =>
      rewardHistory.map((p) => ({
        step: p.step,
        reward: p.reward,
        healthPct: Number((p.health * 100).toFixed(2)),
      })),
    [rewardHistory]
  );

  return (
    <>
      <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-widest">
        Performance Trend
      </h3>
      <div className="bg-[#1E293B]/60 backdrop-blur-xl border border-white/10 rounded-xl p-6">
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
            {hasData && (
              <span className="font-mono text-[11px] px-2 py-0.5 rounded border text-on-surface-variant bg-white/5 border-white/10">
                {rewardHistory.length} step{rewardHistory.length !== 1 ? "s" : ""}
              </span>
            )}
          </div>
          <span className="material-symbols-outlined text-outline flex-shrink-0">show_chart</span>
        </div>

        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-2">
            <div className="w-4 h-0.5 bg-gradient-to-r from-blue-500 to-cyan-400 rounded-full" />
            <span className="text-caption text-on-surface-variant">Reward</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 border-t border-dashed border-emerald-400" />
            <span className="text-caption text-on-surface-variant">Health %</span>
          </div>
        </div>

        <div className="h-64 w-full">
          {!hasData ? (
            <div className="flex flex-col items-center justify-center w-full h-full text-on-surface-variant/40 gap-2">
              <span className="material-symbols-outlined text-3xl">timeline</span>
              <p className="font-caption text-caption">No simulation data yet</p>
              <p className="font-caption text-[10px] text-on-surface-variant/30">
                POST /reset -&gt; run actions to see curves update
              </p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={data}
                margin={{ top: 10, right: 12, left: 6, bottom: 36 }}
              >
                <defs>
                  <linearGradient id="rewardAreaGrad" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#22D3EE" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#3B82F6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="rewardLineGrad" x1="0" x2="1" y1="0" y2="0">
                    <stop offset="0%" stopColor="#3B82F6" />
                    <stop offset="100%" stopColor="#22D3EE" />
                  </linearGradient>
                </defs>

                <CartesianGrid stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                <XAxis
                  dataKey="step"
                  tick={{ fill: "#94A3B8", fontSize: 11, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(255,255,255,0.15)" }}
                />
                <YAxis
                  yAxisId="reward"
                  tick={{ fill: "#94A3B8", fontSize: 11, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(255,255,255,0.15)" }}
                  domain={["auto", "auto"]}
                />
                <YAxis
                  yAxisId="health"
                  orientation="right"
                  domain={[0, 100]}
                  tickFormatter={(v) => `${v}%`}
                  tick={{ fill: "#86EFAC", fontSize: 11, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(52,211,153,0.35)" }}
                />
                <Tooltip
                  cursor={{ stroke: "rgba(148,163,184,0.4)", strokeWidth: 1 }}
                  content={<CustomTooltip />}
                />

                <Area
                  yAxisId="reward"
                  type="monotone"
                  dataKey="reward"
                  name="Reward"
                  fill="url(#rewardAreaGrad)"
                  stroke="none"
                />
                <Line
                  yAxisId="reward"
                  type="monotone"
                  dataKey="reward"
                  name="Reward"
                  stroke="url(#rewardLineGrad)"
                  strokeWidth={1.6}
                  dot={false}
                  activeDot={{ r: 4, stroke: "#fff", strokeWidth: 1, fill: "#22D3EE" }}
                />
                <Line
                  yAxisId="health"
                  type="monotone"
                  dataKey="healthPct"
                  name="Health %"
                  stroke="#34D399"
                  strokeDasharray="4 4"
                  strokeWidth={1.1}
                  dot={false}
                />

                <Brush
                  dataKey="step"
                  height={20}
                  stroke="#38BDF8"
                  travellerWidth={8}
                  fill="rgba(15,23,42,0.6)"
                  tickFormatter={(v) => `S${v}`}
                />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </>
  );
}
