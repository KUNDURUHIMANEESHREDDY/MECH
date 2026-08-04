import React from 'react';
import { colors } from '../design/tokens/colors';

interface LayerPanelProps {
  layerIdx: number;
  numHeads: number;
}

export const LayerPanel: React.FC<LayerPanelProps> = ({ layerIdx, numHeads }) => {
  const cardBg = colors.surfacePearl;
  const border = colors.border;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 12 }}>
      {/* Layer Metadata Header */}
      <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
        <div style={{ fontWeight: 700, fontSize: 13, color: colors.primary }}>Layer {layerIdx} Metadata</div>
        <div style={{ color: colors.inkMuted48, marginTop: 2 }}>Sublayers: Attention ({numHeads} heads) + MLP (GELU) + Residual Stream</div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
        {/* Attention Summary */}
        <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 700, color: colors.warning, marginBottom: 4 }}>Attention Summary</div>
          <div>Heads: {numHeads}</div>
          <div>Avg Entropy: 2.14</div>
          <div>Max Attn Weight: 0.94</div>
        </div>

        {/* MLP Summary */}
        <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 700, color: colors.success, marginBottom: 4 }}>MLP Summary</div>
          <div>Activation: GELU</div>
          <div>Sparsity: 42.5% zeroed</div>
          <div>Mean Activation: 0.86</div>
        </div>

        {/* Residual Summary */}
        <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 700, color: colors.pink, marginBottom: 4 }}>Residual Summary</div>
          <div>L2 Norm: 24.12</div>
          <div>Norm Delta: +1.84</div>
          <div>Stream Variance: 0.42</div>
        </div>
      </div>
    </div>
  );
};
