export default function PerformanceTrend() {
  return (
    <>
      <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase">Performance Trend</h3>
      <div className="bg-[#1E293B]/60 backdrop-blur-xl border border-white/10 rounded-xl p-6">
        <div className="flex justify-between items-center mb-6">
          <h4 className="font-h2 text-body-lg text-on-surface truncate pr-2">System Reward Curve</h4>
          <span className="material-symbols-outlined text-outline flex-shrink-0">show_chart</span>
        </div>
        
        <div className="flex items-center gap-4 mb-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-[#4D8EFF] rounded-full"></div>
            <span className="text-caption text-on-surface-variant">Reward</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 border-t border-dashed border-[#4ade80]"></div>
            <span className="text-caption text-on-surface-variant">Health</span>
          </div>
        </div>
        
        <div className="h-48 w-full relative flex items-end">
          <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-caption text-outline-variant py-2 pr-2">
            <span>100</span>
            <span>50</span>
            <span>0</span>
          </div>
          <div className="ml-8 w-full h-full relative overflow-hidden">
            <div className="absolute inset-0 flex flex-col justify-between py-4">
              <div className="w-full h-[1px] bg-white/5"></div>
              <div className="w-full h-[1px] bg-white/5"></div>
              <div className="w-full h-[1px] bg-white/5"></div>
            </div>
            <svg className="absolute w-full h-full" preserveAspectRatio="none" viewBox="0 0 100 50">
              <defs>
                <linearGradient id="indigoGradient" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#4D8EFF" stopOpacity="0.5"></stop>
                  <stop offset="100%" stopColor="#4D8EFF" stopOpacity="0"></stop>
                </linearGradient>
                <linearGradient id="greenGradient" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#4ade80" stopOpacity="0.3"></stop>
                  <stop offset="100%" stopColor="#4ade80" stopOpacity="0"></stop>
                </linearGradient>
              </defs>
              
              <path d="M0,50 L0,40 Q10,42 20,38 T40,41 T60,39 T80,43 T100,40 L100,50 Z" fill="url(#greenGradient)"></path>
              <path d="M0,40 Q10,42 20,38 T40,41 T60,39 T80,43 T100,40" fill="none" stroke="#4ade80" strokeDasharray="2 2" strokeWidth="1"></path>
              
              <path d="M0,50 L0,30 Q10,35 20,25 T40,15 T60,20 T80,5 T100,10 L100,50 Z" fill="url(#indigoGradient)"></path>
              <path d="M0,30 Q10,35 20,25 T40,15 T60,20 T80,5 T100,10" fill="none" stroke="#4D8EFF" strokeWidth="2"></path>
            </svg>
            <div className="absolute right-[0%] top-[20%] w-3 h-3 bg-background rounded-full border-2 border-primary shadow-[0_0_10px_rgba(77,142,255,0.6)]"></div>
          </div>
        </div>
      </div>
    </>
  );
}
