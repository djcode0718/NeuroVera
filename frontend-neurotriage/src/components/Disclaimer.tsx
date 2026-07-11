import { AlertTriangle } from 'lucide-react';

export function Disclaimer() {
  return (
    <div
      style={{
        backgroundColor: 'var(--color-warning-bg)',
        border: '1px solid var(--color-warning-border)',
      }}
      className="flex items-start gap-3 rounded-xl px-4 py-3 mt-6"
    >
      <AlertTriangle
        className="shrink-0 mt-0.5"
        size={15}
        style={{ color: 'var(--color-warning)' }}
      />
      <p className="text-sm leading-relaxed" style={{ color: 'var(--color-warning)' }}>
        <strong className="font-semibold">Research prototype.</strong>{' '}
        Not a diagnostic tool. Built on a non-clinically-validated classifier.{' '}
        Not for use in real patient care.
      </p>
    </div>
  );
}
