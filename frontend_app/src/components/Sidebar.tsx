import { IncidentState } from '@/hooks/useEnvironment';

interface SidebarProps {
  collapsed: boolean;
  state?: IncidentState | null;
  onReset?: () => void;
}

export default function Sidebar({ collapsed, state, onReset }: SidebarProps) {
  return (
    <nav
      className={`bg-surface-container-lowest text-on-surface-variant font-manrope text-[11px] uppercase tracking-widest font-semibold h-screen fixed left-0 top-0 border-r border-outline-variant/10 flex flex-col pt-16 pb-6 z-40 sidebar-transition ${
        collapsed ? 'collapsed-sidebar' : 'w-60'
      }`}
      id="sidebar"
    >
      <div className="px-6 mb-8 whitespace-nowrap overflow-hidden">
        <h1 className="text-sm font-bold text-on-surface font-h2 mb-0.5 tracking-tight">CONTROL CENTER</h1>
        <p className="text-[10px] text-outline normal-case tracking-normal font-medium opacity-70 truncate">
          {state?.episode_id ? `Session: ${state.episode_id.split('-')[0]}` : 'Waiting for connection...'}
        </p>
      </div>
      
      <div className="flex flex-col gap-0.5 w-full flex-grow px-3 overflow-hidden">
        <a className="flex items-center gap-3 px-3 py-2 bg-primary/5 text-primary border-r-2 border-primary rounded hover:bg-white/5 transition-colors duration-150 whitespace-nowrap" href="#">
          <span className="material-symbols-outlined text-base">hub</span>
          <span className="text-[11px]">Service Map</span>
        </a>
        <a className="flex items-center gap-3 px-3 py-2 text-on-surface-variant/70 hover:text-on-surface rounded hover:bg-white/5 transition-colors duration-150 whitespace-nowrap" href="#">
          <span className="material-symbols-outlined text-base">segment</span>
          <span className="text-[11px]">Activity Log</span>
        </a>
        <a className="flex items-center gap-3 px-3 py-2 text-on-surface-variant/70 hover:text-on-surface rounded hover:bg-white/5 transition-colors duration-150 whitespace-nowrap" href="#">
          <span className="material-symbols-outlined text-base">monitoring</span>
          <span className="text-[11px]">Performance</span>
        </a>
      </div>
      
      <div className="px-4 mt-auto mb-6 overflow-hidden">
        <button 
          onClick={onReset}
          className="w-full bg-error/10 border border-error/20 text-error text-[10px] font-semibold tracking-wider uppercase py-2 rounded transition-colors hover:bg-error/20 whitespace-nowrap flex justify-center items-center gap-2"
        >
          <span className="material-symbols-outlined text-sm">refresh</span>
          Restart Sim
        </button>
      </div>
    </nav>
  );
}
