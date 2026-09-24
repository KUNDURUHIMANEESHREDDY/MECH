import React from 'react';

interface TokenPanelProps {
  tokens: string[];
  tokenIds: number[];
  selectedTokenIdx: number | null;
  onSelectToken: (idx: number) => void;
  residNorms?: number[] | null;
}

export const TokenPanel: React.FC<TokenPanelProps> = ({
  tokens,
  tokenIds,
  selectedTokenIdx,
  onSelectToken,
  residNorms,
}) => {
  if (!tokens || tokens.length === 0) {
    return <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No token data available.</div>;
  }

  const activeIdx = selectedTokenIdx ?? 0;
  const norm = residNorms && residNorms[activeIdx] !== undefined
    ? residNorms[activeIdx].toFixed(2)
    : '—';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {tokens.map((t, idx) => (
          <button
            key={idx}
            onClick={() => onSelectToken(idx)}
            style={{
              background: idx === activeIdx ? 'var(--accent)' : 'var(--bg-elev-2)',
              color: idx === activeIdx ? 'var(--bg)' : 'var(--text)',
              border: 'none',
              borderRadius: 4,
              padding: '4px 8px',
              cursor: 'pointer',
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            {t} <span style={{ opacity: 0.6, fontSize: 10 }}>#{idx}</span>
          </button>
        ))}
      </div>

      <div style={{ background: 'var(--bg-elev-2)', padding: 10, borderRadius: 6, fontSize: 12 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>Selected Token: "{tokens[activeIdx]}" (Index {activeIdx})</div>
        <div style={{ color: 'var(--text-muted)' }}>
          Token ID: {tokenIds[activeIdx] ?? '—'} · Position {activeIdx + 1}/{tokens.length} · Residual L2 (selected layer): {norm}
        </div>
      </div>
    </div>
  );
};
