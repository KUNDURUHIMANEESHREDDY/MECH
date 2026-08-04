import React from 'react';

interface Props {
  tokens: string[];
  tokenIds: number[];
  selectedToken: number | null;
  onHoverToken: (i: number | null) => void;
}

export function TokenViewer({ tokens, tokenIds, selectedToken, onHoverToken }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontFamily: 'monospace', fontSize: 12 }}>
      <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        {tokens.map((t, i) => (
          <span
            key={i}
            onMouseEnter={() => onHoverToken(i)}
            onMouseLeave={() => onHoverToken(null)}
            style={{
              padding: '2px 6px',
              borderRadius: 4,
              background: selectedToken === i ? 'var(--accent)' : 'var(--bg-elev-2)',
              color: selectedToken === i ? 'var(--bg)' : 'var(--text)',
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
          >
            {t}
          </span>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap', color: 'var(--text-muted)' }}>
        {tokenIds.map((id, i) => (
          <span
            key={i}
            style={{
              padding: '2px 6px',
              borderRadius: 4,
              background: selectedToken === i ? 'var(--accent-soft)' : 'transparent',
            }}
          >
            {id}
          </span>
        ))}
      </div>
    </div>
  );
}
