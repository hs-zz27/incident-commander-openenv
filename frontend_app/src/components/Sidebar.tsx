export default function Sidebar({ collapsed }: { collapsed: boolean }) {
  return (
    <nav
      className={`bg-surface-container-lowest text-on-surface-variant font-manrope text-[11px] uppercase tracking-widest font-semibold h-screen fixed left-0 top-0 border-r border-outline-variant/10 flex flex-col pt-16 pb-6 z-40 sidebar-transition ${
        collapsed ? 'collapsed-sidebar' : 'w-60'
      }`}
      id="sidebar"
    >
      <div className="px-6 mb-8 whitespace-nowrap overflow-hidden">
        <h1 className="text-sm font-bold text-on-surface font-h2 mb-0.5 tracking-tight">CONTROL CENTER</h1>
        <p className="text-[10px] text-outline normal-case tracking-normal font-medium opacity-70">Active Session</p>
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
        <a className="flex items-center gap-3 px-3 py-2 text-on-surface-variant/70 hover:text-on-surface rounded hover:bg-white/5 transition-colors duration-150 whitespace-nowrap" href="#">
          <span className="material-symbols-outlined text-base">groups</span>
          <span className="text-[11px]">Team Status</span>
        </a>
      </div>
      
      <div className="px-4 mt-auto mb-6 overflow-hidden">
        <button className="w-full bg-primary/10 border border-primary/20 text-primary text-[10px] font-semibold tracking-wider uppercase py-2 rounded transition-colors hover:bg-primary/20 whitespace-nowrap">
          System Health
        </button>
      </div>
      
      <div className="flex flex-col gap-0.5 w-full px-3 border-t border-outline-variant/10 pt-4 overflow-hidden">
        <a className="flex items-center gap-3 px-3 py-2 text-on-surface-variant/70 hover:text-on-surface rounded hover:bg-white/5 transition-colors duration-150 whitespace-nowrap" href="#">
          <span className="material-symbols-outlined text-base">settings</span>
          <span className="text-[11px]">Settings</span>
        </a>
        <a className="flex items-center gap-3 px-3 py-2 text-on-surface-variant/70 hover:text-on-surface rounded hover:bg-white/5 transition-colors duration-150 whitespace-nowrap" href="#">
          <span className="material-symbols-outlined text-base">logout</span>
          <span className="text-[11px]">Log Out</span>
        </a>
      </div>
    </nav>
  );
}
