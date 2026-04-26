"use client";
import { IncidentState, ServiceState } from '@/hooks/useEnvironment';

interface SidebarProps {
  collapsed: boolean;
  state?: IncidentState | null;
  services?: Record<string, ServiceState>;
}

export default function Sidebar({
  collapsed, state, services,
}: SidebarProps) {
  const serviceList = services ? Object.values(services) : [];
  const hasData = serviceList.length > 0;

  // Average CPU across all services
  const avgCpu = hasData
    ? serviceList.reduce((sum, s) => sum + s.cpu_percent, 0) / serviceList.length
    : 0;

  // Uptime: fraction of services that are NOT down
  const uptime = hasData
    ? (serviceList.filter(s => s.status !== 'down').length / serviceList.length) * 100
    : 100;

  // SVG gauge math
  const circumference = 251.2;
  const cpuOffset = circumference - (circumference * (avgCpu / 100));
  const uptimeOffset = circumference - (circumference * (uptime / 100));
  const cpuColor = avgCpu > 80 ? '#ef4444' : avgCpu > 60 ? '#facc15' : '#4ade80';

  return (
    <nav
      className={`bg-surface-container-lowest text-on-surface-variant font-manrope text-[11px] uppercase tracking-widest font-semibold h-screen fixed left-0 top-0 border-r border-outline-variant/10 flex flex-col pt-16 pb-6 z-40 sidebar-transition ${
        collapsed ? 'collapsed-sidebar' : 'w-48'
      }`}
      id="sidebar"
    >
      <div className="px-6 mb-8 whitespace-nowrap overflow-hidden">
        <h1 className="text-sm font-bold text-on-surface font-h2 mb-0.5 tracking-tight">CONTROL CENTER</h1>
        <p className="text-[10px] text-outline normal-case tracking-normal font-medium opacity-70 truncate">
          {state?.episode_id ? `Session: ${state.episode_id.split('-')[0]}` : 'Waiting for connection...'}
        </p>
      </div>
      
      <div className="flex flex-col gap-0.5 w-full flex-grow px-3 overflow-y-auto overflow-x-hidden">


        {/* CPU Load Gauge */}
        <div className={`bg-surface-container-high/40 backdrop-blur-xl border rounded-xl p-3 flex flex-col items-center transition-all hover:bg-surface-container-high/60 mb-3 ${avgCpu > 80 ? 'border-error/30' : avgCpu > 60 ? 'border-yellow-500/30' : 'border-primary/20'}`}>
          <div className="flex items-center gap-1 mb-3 group relative cursor-help">
            <h4 className="font-label-caps text-label-caps text-on-surface-variant text-[9px] whitespace-nowrap">CPU Load</h4>
            <span className="material-symbols-outlined text-[10px] text-on-surface-variant/40 group-hover:text-primary transition-colors">info</span>
            {/* Tooltip */}
            <div className="absolute left-0 top-full mt-1 hidden group-hover:block w-32 p-2 bg-surface-container-highest border border-outline-variant/20 rounded shadow-lg text-[9px] text-on-surface-variant z-50 font-normal normal-case leading-tight text-center">
              Average processing load across all active core systems.
            </div>
          </div>
          <div className="relative w-16 h-16 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" fill="none" r="40" stroke="rgba(255,255,255,0.05)" strokeDasharray={circumference.toString()} strokeDashoffset="0" strokeWidth="8"></circle>
              <circle cx="50" cy="50" fill="none" r="40" stroke={cpuColor} strokeDasharray={circumference.toString()} strokeDashoffset={cpuOffset} strokeLinecap="round" strokeWidth="8" className="transition-all duration-700"></circle>
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className="font-display text-sm text-on-surface font-bold">{hasData ? `${avgCpu.toFixed(0)}%` : '—'}</span>
            </div>
          </div>
        </div>

        {/* System Uptime Gauge */}
        <div className="bg-surface-container-high/40 backdrop-blur-xl border border-primary/20 rounded-xl p-3 flex flex-col items-center transition-all hover:bg-surface-container-high/60">
          <div className="flex items-center gap-1 mb-3 group relative cursor-help">
            <h4 className="font-label-caps text-label-caps text-on-surface-variant text-[9px] whitespace-nowrap">Uptime</h4>
            <span className="material-symbols-outlined text-[10px] text-on-surface-variant/40 group-hover:text-primary transition-colors">info</span>
            {/* Tooltip */}
            <div className="absolute left-0 top-full mt-1 hidden group-hover:block w-32 p-2 bg-surface-container-highest border border-outline-variant/20 rounded shadow-lg text-[9px] text-on-surface-variant z-50 font-normal normal-case leading-tight text-center">
              Percentage of core systems currently operational.
            </div>
          </div>
          <div className="relative w-16 h-16 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" fill="none" r="40" stroke="rgba(255,255,255,0.05)" strokeDasharray={circumference.toString()} strokeDashoffset="0" strokeWidth="8"></circle>
              <circle cx="50" cy="50" fill="none" r="40" stroke="#4ade80" strokeDasharray={circumference.toString()} strokeDashoffset={uptimeOffset} strokeLinecap="round" strokeWidth="8" className="transition-all duration-700"></circle>
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className="font-display text-sm text-on-surface font-bold">{hasData ? `${uptime.toFixed(0)}%` : '—'}</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
