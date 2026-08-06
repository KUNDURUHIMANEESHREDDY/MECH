import React from 'react';

interface PredictionPanelProps {
  tokens: string[];
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({ tokens }) => {
  const cardBg = 'var(--bg-elev-2)';
  const border = 'var(--border)';

  // Mock Top-K Logit Lens predictions across layers
  const topKPredictions = [
    { layer: 0, topToken: 'the', prob: 0.12, entropy: 4.8 },
    { layer: 4, topToken: 'world', prob: 0.35, entropy: 3.2 },
    { layer: 8, topToken: 'world', prob: 0.68, entropy: 1.9 },
    { layer: 11, topToken: 'world', prob: 0.89, entropy: 0.8 },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12 }}>
      <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
        <div style={{ fontWeight: 700, fontSize: 13, color: 'var(--purple)' }}>Prediction Inspector & Logit Lens</div>
        <div style={{ color: 'var(--text-muted)', marginTop: 2 }}>Tracks intermediate token predictions & Shannon entropy layer by layer</div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {topKPredictions.map(p => (
          <div
            key={p.layer}
            style={{
              display: 'flex',
              justify: 'space-between',
              alignItems: 'center',
              background: cardBg,
              padding: '8px 12px',
              borderRadius: 6,
              border: `1px solid ${border}`,
            }}
          >
            <div style={{ fontWeight: 600 }}>Layer {p.layer} Logit Lens</div>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <span style={{ background: 'var(--purple)', color: 'var(--bg)', padding: '2px 8px', borderRadius: 4, fontWeight: 700 }}>
                "{p.topToken}" ({(p.prob * 100).toFixed(0)}%)
              </span>
              <span style={{ color: 'var(--text-muted)' }}>Entropy: {p.entropy} bits</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
