interface HeaderProps {
  onToggleSidebar: () => void;
  isConnected?: boolean;
  taskName?: string;
}

export default function Header({ onToggleSidebar, isConnected = false, taskName }: HeaderProps) {
  return (
    <header className="bg-surface/80 backdrop-blur-md text-on-surface-variant font-manrope text-[12px] font-medium tracking-tight top-0 z-50 border-b border-outline-variant/10 flex justify-between items-center w-full px-6 h-14 sticky">
      <div className="flex items-center gap-4">
        <button 
          onClick={onToggleSidebar}
          className="text-on-surface-variant/60 hover:text-on-surface p-1.5 rounded transition-colors flex items-center justify-center"
          id="sidebar-toggle"
        >
          <span className="material-symbols-outlined text-xl">menu</span>
        </button>
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-on-surface tracking-tight">Incident Commander</span>
          {taskName && (
            <span className="text-xs font-mono bg-surface-container-high text-primary px-2 py-0.5 rounded border border-primary/20">
              {taskName}
            </span>
          )}
          <div title={isConnected ? "Connected to Backend" : "Disconnected"} className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 shadow-[0_0_5px_rgba(34,197,94,0.5)]' : 'bg-red-500'}`}></div>
        </div>
      </div>
      
      <div className="flex items-center gap-4">
        <div className="relative hidden sm:block">
          <span className="material-symbols-outlined absolute left-2.5 top-1/2 -translate-y-1/2 text-outline text-base">search</span>
          <input 
            type="text" 
            placeholder="Search systems..." 
            className="bg-surface-container-low border border-outline-variant/20 rounded-md py-1 pl-8 pr-4 text-[12px] text-on-surface focus:outline-none focus:border-primary/50 w-56 placeholder-outline/50 font-body-md" 
          />
        </div>
        
        <div className="flex items-center gap-1.5">
          <button className="text-on-surface-variant/60 hover:text-on-surface hover:bg-white/5 transition-all duration-200 rounded p-1.5 flex items-center justify-center">
            <span className="material-symbols-outlined text-lg">notifications</span>
          </button>
          <button className="text-on-surface-variant/60 hover:text-on-surface hover:bg-white/5 transition-all duration-200 rounded p-1.5 flex items-center justify-center">
            <span className="material-symbols-outlined text-lg">history</span>
          </button>
          <div className="h-7 w-7 rounded bg-surface-variant overflow-hidden border border-outline-variant/20 ml-1.5">
            <img 
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuBaMVxndMBPMC54iJT14YPkxEHYVMa-lXXhwgN5qmmJC6nw9O53tCk2lAjKrQZxfE6wfH-nxyT_P_AOOcCVYljWJymraYszy9CkXubpALngKR4DyNcUd9LF5Gtrt0iNsIi8FJuJVTS-QpjFnTHRveFL7zxKep_8EI8pcAG2AR3ah41drjI3jboeRKIZo0URqp3zp4Hz-2DhoGau4Yosq7bmzpFYyW7EIfaYqam_4jGcF2LXmHGoJOrEXUpS7JTW7qfFsmn_68vEWxas" 
              alt="Commander Profile" 
              className="w-full h-full object-cover" 
            />
          </div>
        </div>
      </div>
    </header>
  );
}
