import { ServiceState } from '@/hooks/useEnvironment';

const SERVICE_META: Record<string, { icon: string; label: string }> = {
  database:     { icon: 'database',               label: 'Database' },
  cache:        { icon: 'memory',                 label: 'Cache' },
  auth:         { icon: 'shield_person',          label: 'Auth' },
  notification: { icon: 'campaign',               label: 'Notification' },
  payments:     { icon: 'payments',               label: 'Payments' },
  checkout:     { icon: 'shopping_cart_checkout',  label: 'Checkout' },
};

function statusStyle(status: string) {
  switch (status) {
    case 'healthy':
      return {
        badge: 'text-primary bg-primary/20',
        icon:  'bg-primary/10 text-primary',
        row:   'bg-white/5 border-transparent hover:border-white/10',
        text:  'text-on-surface font-medium',
        label: 'Healthy',
      };
    case 'degraded':
      return {
        badge: 'text-tertiary bg-tertiary/20',
        icon:  'bg-tertiary/10 text-tertiary',
        row:   'bg-tertiary/5 border-tertiary/20',
        text:  'text-on-surface font-medium',
        label: 'Degraded',
      };
    case 'down':
      return {
        badge: 'text-error bg-error/20',
        icon:  'bg-error/20 text-error',
        row:   'bg-error/10 border-error/30 glow-amber',
        text:  'text-error font-semibold',
        label: 'Critical',
      };
    default:
      return {
        badge: 'text-outline bg-white/10',
        icon:  'bg-white/5 text-outline',
        row:   'bg-white/5 border-transparent',
        text:  'text-on-surface-variant',
        label: 'Unknown',
      };
  }
}

interface ServiceMapProps {
  services: Record<string, ServiceState>;
}

export default function ServiceMap({ services }: ServiceMapProps) {
  const serviceNames = Object.keys(SERVICE_META);
  const hasData = Object.keys(services).length > 0;

  return (
    <div className="flex flex-col gap-stack-md h-full">
      <h3 className="font-label-caps text-label-caps text-on-surface-variant uppercase">Core Systems</h3>
      <div className="bg-[#1E293B]/60 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden flex flex-col gap-2 p-3 h-full">
        {serviceNames.map((name) => {
          const svc = services[name];
          const meta = SERVICE_META[name];
          const style = statusStyle(svc?.status || 'unknown');

          return (
            <div
              key={name}
              className={`p-3 rounded-lg flex items-center justify-between border transition-colors gap-3 ${style.row}`}
            >
              <div className="flex items-center gap-3 min-w-0 flex-1">
                <div className={`w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center ${style.icon}`}>
                  <span className="material-symbols-outlined text-sm">{meta.icon}</span>
                </div>
                <div className="flex flex-col min-w-0">
                  <span className={`font-body-md text-body-md truncate ${style.text}`}>{meta.label}</span>
                  {hasData && svc && (
                    <span className="font-caption text-[10px] text-on-surface-variant/60 truncate">
                      CPU {svc.cpu_percent.toFixed(0)}% · {svc.latency_ms.toFixed(0)}ms · Err {(svc.error_rate * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              </div>
              <span className={`font-caption text-[11px] font-bold px-2 py-0.5 rounded-full flex-shrink-0 whitespace-nowrap ${style.badge}`}>
                {style.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
