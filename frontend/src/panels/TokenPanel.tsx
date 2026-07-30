import React from 'react';

interface TokenPanelProps {
  tokens: string[];
  selectedTokenIdx: number | null;
  onSelectToken: (idx: number) => void;
  darkMode: boolean;
}

export const TokenPanel: React.FC<TokenPanelProps> = ({ tokens, selectedTokenIdx, onSelectToken, darkMode }) => {
  if (!tokens || tokens.length === 0) {
    return <div style={{ color: '#888', fontSize: 12 }}>No token data available.</div>;
  }

  const activeIdx = selectedTokenIdx ?? 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {tokens.map((t, idx) => (
          <button
            key={idx}
            onClick={() => onSelectToken(idx)}
            style={{
              background: idx === activeIdx ? '#3b82f6' : (darkMode ? '#2a2a3a' : '#e0e0e0'),
              color: idx === activeIdx ? '#fff' : (darkMode ? '#ddd' : '#222'),
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

      <div style={{ background: darkMode ? '#12121a' : '#f8f8fc', padding: 10, borderRadius: 6, fontSize: 12 }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>Selected Token: "{tokens[activeIdx]}" (Index {activeIdx})</div>
        <div style={{ color: '#888' }}>Embedding Norm: 14.82 | Residual Delta: 2.14 | Next Token Logit: 18.9</div>
      </div>
    </div>
  );
};
