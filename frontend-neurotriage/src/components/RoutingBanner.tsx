import { CheckCircle, AlertTriangle, AlertOctagon } from 'lucide-react';
import type { AnalysisResult } from '../types/api';

type Routing = 'urgent' | 'needs-review' | 'auto-clear';

interface RoutingConfig {
  label: string;
  description: string;
  icon: React.ReactNode;
  bgColor: string;
  borderColor: string;
  textColor: string;
  accentColor: string;
}

export function getRoutingConfig(routing: Routing): RoutingConfig {
  switch (routing) {
    case 'urgent':
      return {
        label: 'Urgent',
        description: 'Immediate radiologist review required',
        icon: <AlertOctagon size={28} />,
        bgColor: 'var(--color-danger-bg)',
        borderColor: 'var(--color-danger-border)',
        textColor: 'var(--color-danger)',
        accentColor: '#DC2626',
      };
    case 'needs-review':
      return {
        label: 'Needs Review',
        description: 'Clinical review recommended before clearance',
        icon: <AlertTriangle size={28} />,
        bgColor: 'var(--color-warning-bg)',
        borderColor: 'var(--color-warning-border)',
        textColor: 'var(--color-warning)',
        accentColor: '#D97706',
      };
    case 'auto-clear':
      return {
        label: 'Auto-Clear',
        description: 'High-confidence no-tumor classification',
        icon: <CheckCircle size={28} />,
        bgColor: 'var(--color-success-bg)',
        borderColor: 'var(--color-success-border)',
        textColor: 'var(--color-success)',
        accentColor: '#16A34A',
      };
  }
}

interface RoutingBadgeProps {
  routing: Routing;
  size?: 'sm' | 'md';
}

export function RoutingBadge({ routing, size = 'sm' }: RoutingBadgeProps) {
  const config = getRoutingConfig(routing);
  const px = size === 'sm' ? 'px-2.5 py-0.5' : 'px-3 py-1';
  const text = size === 'sm' ? 'text-xs' : 'text-sm';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium ${px} ${text}`}
      style={{
        backgroundColor: config.bgColor,
        color: config.textColor,
        border: `1px solid ${config.borderColor}`,
      }}
    >
      {size === 'md' && (
        <span className="flex-shrink-0" style={{ width: 12, height: 12 }}>
          {routing === 'urgent' ? <AlertOctagon size={12} /> : routing === 'needs-review' ? <AlertTriangle size={12} /> : <CheckCircle size={12} />}
        </span>
      )}
      {config.label}
    </span>
  );
}

interface RoutingBannerProps {
  result: Pick<AnalysisResult, 'routing' | 'run_id'>;
}

export function RoutingBanner({ result }: RoutingBannerProps) {
  const config = getRoutingConfig(result.routing);

  return (
    <div
      className="relative rounded-2xl p-5 overflow-hidden"
      style={{
        background: `linear-gradient(135deg, ${config.bgColor}, ${config.bgColor}dd)`,
        border: `1px solid ${config.borderColor}`,
        boxShadow: `0 4px 24px -4px ${config.accentColor}20`,
      }}
    >
      {/* Subtle decorative circle */}
      <div
        className="absolute -top-6 -right-6 w-32 h-32 rounded-full opacity-10"
        style={{ backgroundColor: config.accentColor }}
      />

      {/* Run ID tag — top right */}
      <div className="absolute top-4 right-4">
        <span
          className="text-xs px-2 py-1 rounded-md"
          style={{
            fontFamily: 'var(--font-mono)',
            backgroundColor: `${config.accentColor}15`,
            color: config.textColor,
            border: `1px solid ${config.borderColor}`,
          }}
        >
          {result.run_id.slice(0, 8)}…
        </span>
      </div>

      <div className="flex items-start gap-4 pr-24">
        <div style={{ color: config.accentColor }} className="shrink-0 mt-0.5">
          {config.icon}
        </div>
        <div>
          <h2
            className="text-xl font-bold tracking-tight mb-1"
            style={{ fontFamily: 'var(--font-display)', color: config.textColor }}
          >
            {config.label}
          </h2>
          <p className="text-sm" style={{ color: config.textColor, opacity: 0.8 }}>
            {config.description}
          </p>
        </div>
      </div>
    </div>
  );
}
