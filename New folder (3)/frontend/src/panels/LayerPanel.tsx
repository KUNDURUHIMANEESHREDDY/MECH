import React from 'react';

interface LayerPanelProps {
  layerIdx: number;
  numHeads: number;
  darkMode: boolean;
}

export const LayerPanel: React.FC<LayerPanelProps> = ({ layerIdx, numHeads, darkMode }) => {
  const cardBg = darkMode ? '#12121a' : '#f8f8fc';
  const border = darkMode ? '#2a2a3a' : '#e0e0e0';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 12 }}>
      {/* Layer Metadata Header */}
      <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
        <div style={{ fontWeight: 700, fontSize: 13, color: '#3b82f6' }}>Layer {layerIdx} Metadata</div>
        <div style={{ color: '#888', marginTop: 2 }}>Sublayers: Attention ({numHeads} heads) + MLP (GELU) + Residual Stream</div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
        {/* Attention Summary */}
        <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 700, color: '#eab308', marginBottom: 4 }}>Attention Summary</div>
          <div>Heads: {numHeads}</div>
          <div>Avg Entropy: 2.14</div>
          <div>Max Attn Weight: 0.94</div>
        </div>

        {/* MLP Summary */}
        <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 700, color: '#10b981', marginBottom: 4 }}>MLP Summary</div>
          <div>Activation: GELU</div>
          <div>Sparsity: 42.5% zeroed</div>
          <div>Mean Activation: 0.86</div>
        </div>

        {/* Residual Summary */}
        <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 700, color: '#ec4899', marginBottom: 4 }}>Residual Summary</div>
          <div>L2 Norm: 24.12</div>
          <div>Norm Delta: +1.84</div>
          <div>Stream Variance: 0.42</div>
        </div>
      </div>
    </div>
  );
};
