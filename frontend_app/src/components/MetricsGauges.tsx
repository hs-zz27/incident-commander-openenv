import { ServiceState } from '@/hooks/useEnvironment';

interface MetricsGaugesProps {
  services: Record<string, ServiceState>;
}

export default function MetricsGauges({ services }: MetricsGaugesProps) {
  const serviceList = Object.values(services);
  const hasData = serviceList.length > 0;

  // Calculate average CPU across all services
  const avgCpu = hasData
    ? serviceList.reduce((sum, s) => sum + s.cpu_percent, 0) / serviceList.length
    : 0;

  // Calculate uptime: fraction of services that are NOT down
  const uptime = hasData
    ? (serviceList.filter(s => s.status !== 'down').length / serviceList.length) * 100
    : 100;

  // SVG gauge math: circumference = 2*PI*40 ≈ 251.2
  const circumference = 251.2;
  const cpuOffset = circumference - (circumference * (avgCpu / 100));
  const uptimeOffset = circumference - (circumference * (uptime / 100));

  const cpuColor = avgCpu > 80 ? '#ef4444' : avgCpu > 60 ? '#facc15' : '#4ade80';

  return (
    <>
      <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase">Key Metrics</h3>
      <div className="flex flex-col gap-4 h-full">
        {/* CPU Gauge */}
        <div className={`flex-1 bg-surface-container-high/40 backdrop-blur-xl border rounded-xl p-5 flex flex-col items-center justify-center min-w-0 transition-all hover:bg-surface-container-high/60 ${avgCpu > 80 ? 'border-error/30' : avgCpu > 60 ? 'border-yellow-500/30 glow-amber' : 'border-primary/20 glow-indigo'}`}>
          <h4 className="font-label-caps text-label-caps text-on-surface-variant mb-4 truncate w-full text-center">Avg CPU Load</h4>
          <div className="relative w-20 h-20 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" fill="none" r="40" stroke="rgba(255,255,255,0.05)" strokeDasharray={circumference.toString()} strokeDashoffset="0" strokeWidth="8"></circle>
              <circle cx="50" cy="50" fill="none" r="40" stroke={cpuColor} strokeDasharray={circumference.toString()} strokeDashoffset={cpuOffset} strokeLinecap="round" strokeWidth="8" className="transition-all duration-700"></circle>
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className="font-display text-lg sm:text-h2 text-on-surface">{hasData ? `${avgCpu.toFixed(0)}%` : '—'}</span>
            </div>
          </div>
        </div>

        {/* Uptime Gauge */}
        <div className="flex-1 bg-surface-container-high/40 glow-indigo backdrop-blur-xl border border-primary/20 rounded-xl p-5 flex flex-col items-center justify-center min-w-0 transition-all hover:bg-surface-container-high/60">
          <h4 className="font-label-caps text-label-caps text-on-surface-variant mb-4 truncate w-full text-center">System Uptime</h4>
          <div className="relative w-20 h-20 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" fill="none" r="40" stroke="rgba(255,255,255,0.05)" strokeDasharray={circumference.toString()} strokeDashoffset="0" strokeWidth="8"></circle>
              <circle cx="50" cy="50" fill="none" r="40" stroke="#4ade80" strokeDasharray={circumference.toString()} strokeDashoffset={uptimeOffset} strokeLinecap="round" strokeWidth="8" className="transition-all duration-700"></circle>
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className="font-display text-lg sm:text-h2 text-on-surface">{hasData ? `${uptime.toFixed(0)}%` : '—'}</span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
