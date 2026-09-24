import React from 'react';
import { colors } from '../design/tokens/colors';

interface LayerPanelProps {
  layerIdx: number;
  numHeads: number;
  residL2?: number | null;
  mlpMean?: number | null;
  mlpSparsity?: number | null;
  headEntropy?: number | null;
  loading?: boolean;
}

function fmt(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(digits);
}

export const LayerPanel: React.FC<LayerPanelProps> = ({
  layerIdx,
  numHeads,
  residL2,
  mlpMean,
  mlpSparsity,
  headEntropy,
  loading,
}) => {
  const cardBg = colors.surfacePearl;
  const border = colors.border;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 12 }}>
      {/* Layer Metadata Header */}
      <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
        <div style={{ fontWeight: 700, fontSize: 13, color: colors.primary }}>Layer {layerIdx} Metadata</div>
        <div style={{ color: colors.inkMuted48, marginTop: 2 }}>Sublayers: Attention ({numHeads} heads) + MLP (GELU) + Residual Stream</div>
      </div>

      {loading && <div style={{ color: colors.inkMuted48 }}>Measuring live tensors…</div>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
        {/* Attention Summary */}
        <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 700, color: colors.warning, marginBottom: 4 }}>Attention Summary</div>
          <div>Heads: {numHeads}</div>
          <div>Selected-head entropy: {fmt(headEntropy)} nats</div>
        </div>

        {/* MLP Summary */}
        <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 700, color: colors.success, marginBottom: 4 }}>MLP Summary</div>
          <div>Activation: GELU</div>
          <div>Sparsity (|x|&lt;1e-3): {mlpSparsity === null || mlpSparsity === undefined ? '—' : `${(mlpSparsity * 100).toFixed(1)}%`}</div>
          <div>Mean Activation: {fmt(mlpMean, 4)}</div>
        </div>

        {/* Residual Summary */}
        <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 700, color: colors.pink, marginBottom: 4 }}>Residual Summary</div>
          <div>L2 Norm: {fmt(residL2)}</div>
        </div>
      </div>
    </div>
  );
};
