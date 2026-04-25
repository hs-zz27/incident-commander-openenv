export default function ServiceMap() {
  return (
    <div className="flex flex-col gap-stack-md h-full">
      <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase">Core Systems</h3>
      <div className="bg-[#1E293B]/60 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden flex flex-col gap-2 p-3 h-full">
        {/* Database */}
        <div className="p-3 rounded-lg bg-white/5 flex items-center justify-between border border-transparent hover:border-white/10 transition-colors gap-3">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="w-8 h-8 flex-shrink-0 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="material-symbols-outlined text-primary text-sm">database</span>
            </div>
            <span className="font-body-md text-body-md text-on-surface font-medium truncate">Database</span>
          </div>
          <span className="font-caption text-[11px] font-bold text-primary bg-primary/20 px-2 py-0.5 rounded-full flex-shrink-0 whitespace-nowrap">Healthy</span>
        </div>
        
        {/* Cache */}
        <div className="p-3 rounded-lg bg-white/5 flex items-center justify-between border border-transparent hover:border-white/10 transition-colors gap-3">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="w-8 h-8 flex-shrink-0 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="material-symbols-outlined text-primary text-sm">memory</span>
            </div>
            <span className="font-body-md text-body-md text-on-surface font-medium truncate">Cache</span>
          </div>
          <span className="font-caption text-[11px] font-bold text-primary bg-primary/20 px-2 py-0.5 rounded-full flex-shrink-0 whitespace-nowrap">Healthy</span>
        </div>
        
        {/* Auth */}
        <div className="p-3 rounded-lg bg-white/5 flex items-center justify-between border border-transparent hover:border-white/10 transition-colors gap-3">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="w-8 h-8 flex-shrink-0 rounded-full bg-tertiary/10 flex items-center justify-center">
              <span className="material-symbols-outlined text-tertiary text-sm">shield_person</span>
            </div>
            <span className="font-body-md text-body-md text-on-surface font-medium truncate">Auth</span>
          </div>
          <span className="font-caption text-[11px] font-bold text-tertiary bg-tertiary/20 px-2 py-0.5 rounded-full flex-shrink-0 whitespace-nowrap">Degraded</span>
        </div>
        
        {/* Notification */}
        <div className="p-3 rounded-lg bg-white/5 flex items-center justify-between border border-transparent hover:border-white/10 transition-colors gap-3">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="w-8 h-8 flex-shrink-0 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="material-symbols-outlined text-primary text-sm">campaign</span>
            </div>
            <span className="font-body-md text-body-md text-on-surface font-medium truncate">Notification</span>
          </div>
          <span className="font-caption text-[11px] font-bold text-primary bg-primary/20 px-2 py-0.5 rounded-full flex-shrink-0 whitespace-nowrap">Healthy</span>
        </div>
        
        {/* Payments */}
        <div className="p-3 rounded-lg bg-error/10 border border-error/30 flex items-center justify-between shadow-[0_0_15px_rgba(255,180,171,0.05)] gap-3 glow-amber">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="w-8 h-8 flex-shrink-0 rounded-full bg-error/20 flex items-center justify-center">
              <span className="material-symbols-outlined text-error text-sm">payments</span>
            </div>
            <span className="font-body-md text-body-md text-error font-semibold truncate">Payments</span>
          </div>
          <span className="font-caption text-[11px] font-bold text-error bg-error/20 px-2 py-0.5 rounded-full flex-shrink-0 whitespace-nowrap">Critical</span>
        </div>
        
        {/* Checkout */}
        <div className="p-3 rounded-lg bg-white/5 flex items-center justify-between border border-transparent hover:border-white/10 transition-colors gap-3">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="w-8 h-8 flex-shrink-0 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="material-symbols-outlined text-primary text-sm">shopping_cart_checkout</span>
            </div>
            <span className="font-body-md text-body-md text-on-surface font-medium truncate">Checkout</span>
          </div>
          <span className="font-caption text-[11px] font-bold text-primary bg-primary/20 px-2 py-0.5 rounded-full flex-shrink-0 whitespace-nowrap">Healthy</span>
        </div>
      </div>
    </div>
  );
}
