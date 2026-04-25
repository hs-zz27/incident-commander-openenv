export default function ActivityLog() {
  return (
    <div className="flex flex-col gap-stack-md h-full">
      <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase">Activity Log</h3>
      <div className="bg-[#1E293B]/60 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden h-full flex flex-col p-6">
        <div className="flex-1 overflow-y-auto pr-2 space-y-6">
          {/* Log Entry 1 */}
          <div className="relative pl-6 before:content-[''] before:absolute before:left-[11px] before:top-[24px] before:bottom-[-24px] before:w-[2px] before:bg-white/5 last:before:hidden">
            <div className="absolute left-0 top-1 w-6 h-6 rounded-full bg-surface-container flex items-center justify-center border border-error/30">
              <div className="w-2 h-2 rounded-full bg-error"></div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between gap-2">
                <span className="font-body-md text-body-md text-on-surface font-medium truncate">Payment Gateway Timeout</span>
                <span className="font-caption text-caption text-on-surface-variant flex-shrink-0">Just now</span>
              </div>
              <p className="font-caption text-caption text-on-surface-variant">Multiple timeout errors detected from primary payment processor API.</p>
              <div className="flex gap-2 mt-2">
                <span className="font-caption text-caption text-error bg-error/10 px-2 py-1 rounded">Action Required</span>
              </div>
            </div>
          </div>
          
          {/* Log Entry 2 */}
          <div className="relative pl-6 before:content-[''] before:absolute before:left-[11px] before:top-[24px] before:bottom-[-24px] before:w-[2px] before:bg-white/5 last:before:hidden">
            <div className="absolute left-0 top-1 w-6 h-6 rounded-full bg-surface-container flex items-center justify-center border border-white/10">
              <div className="w-2 h-2 rounded-full bg-secondary"></div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between gap-2">
                <span className="font-body-md text-body-md text-on-surface font-medium truncate">Auto-scaling Triggered</span>
                <span className="font-caption text-caption text-on-surface-variant flex-shrink-0">12 mins ago</span>
              </div>
              <p className="font-caption text-caption text-on-surface-variant">Database cluster scaled up by 2 nodes to handle increased read load.</p>
            </div>
          </div>
          
          {/* Log Entry 3 */}
          <div className="relative pl-6 before:content-[''] before:absolute before:left-[11px] before:top-[24px] before:bottom-[-24px] before:w-[2px] before:bg-white/5 last:before:hidden">
            <div className="absolute left-0 top-1 w-6 h-6 rounded-full bg-surface-container flex items-center justify-center border border-tertiary/30">
              <div className="w-2 h-2 rounded-full bg-tertiary"></div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between gap-2">
                <span className="font-body-md text-body-md text-on-surface font-medium truncate">Auth Latency Spike</span>
                <span className="font-caption text-caption text-on-surface-variant flex-shrink-0">45 mins ago</span>
              </div>
              <p className="font-caption text-caption text-on-surface-variant">Authentication requests experiencing &gt;500ms latency.</p>
              <div className="flex gap-2 mt-2">
                <span className="font-caption text-caption text-tertiary bg-tertiary/10 px-2 py-1 rounded">Investigating</span>
              </div>
            </div>
          </div>
          
          {/* Log Entry 4 */}
          <div className="relative pl-6 last:before:hidden">
            <div className="absolute left-0 top-1 w-6 h-6 rounded-full bg-surface-container flex items-center justify-center border border-white/10">
              <div className="w-2 h-2 rounded-full bg-primary"></div>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between gap-2">
                <span className="font-body-md text-body-md text-on-surface font-medium truncate">System Backup Complete</span>
                <span className="font-caption text-caption text-on-surface-variant flex-shrink-0">2 hrs ago</span>
              </div>
              <p className="font-caption text-caption text-on-surface-variant">Routine snapshot of all core databases completed successfully.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
