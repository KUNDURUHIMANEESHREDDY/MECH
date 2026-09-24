import React from 'react';

export type ProvenanceKind = 'offline' | 'seeded' | 'live' | null;

interface ProvenanceBannerProps {
  kind: ProvenanceKind;
  note?: string;
}

/** Never silently fake results: visible whenever data is not live weights. */
export const ProvenanceBanner: React.FC<ProvenanceBannerProps> = ({ kind, note }) => {
  if (kind === null || kind === 'live') return null;
  const offline = kind === 'offline';
  return (
    <div
      data-testid="provenance-banner"
      data-provenance={kind}
      style={{
        background: offline ? 'var(--red-soft, #fee2e2)' : 'var(--yellow-soft, #fef9c3)',
        color: 'var(--text)',
        border: `1px solid ${offline ? 'var(--red, #ef4444)' : 'var(--yellow, #eab308)'}`,
        borderRadius: 8,
        padding: '8px 12px',
        fontSize: 12,
        marginBottom: 12,
      }}
    >
      <strong>{offline ? 'Backend offline' : 'Seeded demo data'}</strong>
      <span style={{ color: 'var(--text-muted)' }}>
        {' — '}
        {offline
          ? 'cannot reach the runtime at localhost:8000; start the backend for live weights.'
          : (note ?? 'torch/transformers unavailable: deterministic stand-ins, not model measurements.')}
      </span>
    </div>
  );
};
