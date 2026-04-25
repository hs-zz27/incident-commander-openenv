export default function MetricsGauges() {
  return (
    <>
      <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase">Key Metrics</h3>
      <div className="flex flex-col gap-4 h-full">
        {/* CPU Gauge */}
        <div className="flex-1 bg-surface-container-high/40 backdrop-blur-xl border rounded-xl p-5 flex flex-col items-center justify-center min-w-0 transition-all hover:bg-surface-container-high/60 border-yellow-500/30 glow-amber">
          <h4 className="font-label-caps text-label-caps text-on-surface-variant mb-4 truncate w-full text-center">Avg CPU Load</h4>
          <div className="relative w-20 h-20 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" fill="none" r="40" stroke="rgba(255,255,255,0.05)" strokeDasharray="251.2" strokeDashoffset="0" strokeWidth="8"></circle>
              <circle cx="50" cy="50" fill="none" r="40" stroke="#facc15" strokeDasharray="251.2" strokeDashoffset="88" strokeLinecap="round" strokeWidth="8"></circle>
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className="font-display text-lg sm:text-h2 text-on-surface">65%</span>
            </div>
          </div>
        </div>

        {/* Uptime Gauge */}
        <div className="flex-1 bg-surface-container-high/40 glow-indigo backdrop-blur-xl border border-primary/20 rounded-xl p-5 flex flex-col items-center justify-center min-w-0 transition-all hover:bg-surface-container-high/60">
          <h4 className="font-label-caps text-label-caps text-on-surface-variant mb-4 truncate w-full text-center">System Uptime</h4>
          <div className="relative w-20 h-20 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" fill="none" r="40" stroke="rgba(255,255,255,0.05)" strokeDasharray="251.2" strokeDashoffset="0" strokeWidth="8"></circle>
              <circle cx="50" cy="50" fill="none" r="40" stroke="#4ade80" strokeDasharray="251.2" strokeDashoffset="0.25" strokeLinecap="round" strokeWidth="8"></circle>
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className="font-display text-lg sm:text-h2 text-on-surface">99.9%</span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
