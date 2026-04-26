"use client";

const TASKS = [
  { value: 'single_service_failure', label: 'Easy: Single Failure', emoji: '🟢' },
  { value: 'cascading_failure', label: 'Medium: Cascading', emoji: '🟡' },
  { value: 'hidden_root_cause', label: 'Hard: Hidden Root', emoji: '🟠' },
  { value: 'chaos_cascade', label: 'Hard: Chaos Cascade', emoji: '🔴' },
  { value: 'multi_root_cause', label: 'Expert: Multi-Root', emoji: '💀' },
  { value: 'random_incident', label: 'Random Incident', emoji: '🎲' },
];

interface HeaderProps {
  onToggleSidebar: () => void;
  isConnected?: boolean;
  taskName?: string;
  selectedTask?: string;
  onSelectedTaskChange?: (taskName: string) => void;
  chaosMode?: boolean;
  onChaosModeChange?: (enabled: boolean) => void;
  isSimRunning?: boolean;
  isStarting?: boolean;
  onStartSim?: () => void;
  onStopSim?: () => void;
}

export default function Header({
  onToggleSidebar,
  isConnected = false,
  taskName,
  selectedTask,
  onSelectedTaskChange,
  chaosMode,
  onChaosModeChange,
  isSimRunning,
  isStarting,
  onStartSim,
  onStopSim,
}: HeaderProps) {
  const localSelectedTask = selectedTask ?? 'random_incident';
  const localChaosMode = chaosMode ?? true;

  return (
    <header className="bg-surface/80 backdrop-blur-md text-on-surface-variant font-manrope text-[13px] font-medium tracking-tight top-0 z-50 border-b border-outline-variant/10 flex justify-between items-center w-full px-6 h-14 sticky">
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
      
      <div className="flex items-center gap-3">
        {/* Scenario Dropdown */}
        <select
          id="header-task-selector"
          value={localSelectedTask}
          onChange={(e) => onSelectedTaskChange?.(e.target.value)}
          disabled={isSimRunning}
          className="bg-surface-container text-on-surface text-[12px] font-medium py-1.5 px-2.5 rounded-md border border-outline-variant/20 focus:border-primary focus:outline-none transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {TASKS.map((t) => (
            <option key={t.value} value={t.value}>
              {t.emoji} {t.label}
            </option>
          ))}
        </select>

        {/* Chaos Mode Radio Toggle */}
        <div className="flex items-center bg-surface-container rounded-md border border-outline-variant/20 overflow-hidden">
          <button
            onClick={() => onChaosModeChange?.(false)}
            disabled={isSimRunning}
            className={`text-[11px] font-semibold tracking-wider uppercase py-1.5 px-3 transition-all duration-200 disabled:opacity-50 ${
              !localChaosMode
                ? 'bg-blue-500/15 text-blue-400'
                : 'text-on-surface-variant/50 hover:text-on-surface-variant'
            }`}
          >
            Normal
          </button>
          <div className="w-px h-5 bg-outline-variant/20" />
          <button
            onClick={() => onChaosModeChange?.(true)}
            disabled={isSimRunning}
            className={`text-[11px] font-semibold tracking-wider uppercase py-1.5 px-3 transition-all duration-200 flex items-center gap-1 disabled:opacity-50 ${
              localChaosMode
                ? 'bg-error/15 text-error'
                : 'text-on-surface-variant/50 hover:text-on-surface-variant'
            }`}
          >
            ⚡ Chaos
          </button>
        </div>

        {/* Single Start/Stop Sim Button */}
        <button 
          id="header-sim-btn"
          onClick={isSimRunning ? onStopSim : onStartSim}
          disabled={isStarting}
          className={`text-[11px] font-semibold tracking-wider uppercase py-1.5 px-4 rounded-md transition-all duration-300 whitespace-nowrap flex items-center gap-1.5 disabled:opacity-60 ${
            isSimRunning
              ? 'bg-error/15 border border-error/30 text-error hover:bg-error/25'
              : 'bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/25'
          }`}
        >
          {isStarting ? (
            <div className="w-3.5 h-3.5 border-2 border-emerald-400/30 border-t-emerald-400 rounded-full animate-spin" />
          ) : (
            <span className="material-symbols-outlined text-sm">
              {isSimRunning ? 'stop_circle' : 'play_arrow'}
            </span>
          )}
          {isStarting ? 'Starting…' : isSimRunning ? 'Stop Sim' : 'Start Sim'}
        </button>
      </div>
    </header>
  );
}
