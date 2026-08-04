import React, { useMemo } from 'react';
import { colors } from '../../../design/tokens/colors';

interface Props {
  tokens: string[];
  activations: number[];
  maxTokens?: number;
}

const ROW_H = 20;
const GAP = 3;
const LABEL_W = 50;
const BAR_MAX_W = 140;

export function TokenActivationSpectrum({ tokens, activations, maxTokens = 50 }: Props) {
  const limited = useMemo(() => {
    const paired = tokens.map((t, i) => ({ token: t, activation: activations[i] ?? 0, idx: i }));
    paired.sort((a, b) => Math.abs(b.activation) - Math.abs(a.activation));
    return paired.slice(0, maxTokens);
  }, [tokens, activations, maxTokens]);

  const maxAbs = useMemo(() => {
    const m = Math.max(...limited.map(p => Math.abs(p.activation)), 0.01);
    return m;
  }, [limited]);

  const H = limited.length * (ROW_H + GAP);
  const W = LABEL_W + BAR_MAX_W + 60;

  return (
    <div style={{ fontSize: 11 }}>
      <div style={{ fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
        Token Activation Spectrum
      </div>
      <svg width={W} height={H} style={{ display: 'block', borderRadius: 4 }}>
        {limited.map((p, i) => {
          const y = i * (ROW_H + GAP);
          const barW = (Math.abs(p.activation) / maxAbs) * BAR_MAX_W;
          const isPositive = p.activation >= 0;
          const r = isPositive ? Math.round(60 + (1 - p.activation / maxAbs) * 195) : Math.round(60 + (1 + p.activation / maxAbs) * 195);
          const g = isPositive ? Math.round(40 + (p.activation / maxAbs) * 80) : Math.round(40 + (-p.activation / maxAbs) * 80);
          const b = 200;
          return (
            <g key={p.idx}>
              <text x={LABEL_W - 4} y={y + ROW_H / 2} fill={colors.inkMuted48} fontSize="9" textAnchor="end" dominantBaseline="middle">
                {p.token.length > 10 ? p.token.slice(0, 10) + '…' : p.token}
              </text>
              <rect x={LABEL_W} y={y} width={BAR_MAX_W} height={ROW_H} fill={colors.dividerSoft} rx={2} />
              <rect x={LABEL_W} y={y} width={barW} height={ROW_H} fill={`rgb(${r},${g},${b})`} rx={2} />
              <text x={LABEL_W + barW + 4} y={y + ROW_H / 2} fill={colors.bodyMuted} fontSize="9" dominantBaseline="middle">
                {p.activation.toFixed(3)}
              </text>
            </g>
          );
        })}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 10, color: colors.inkMuted48 }}>
        <span>Showing {limited.length} of {tokens.length} tokens</span>
        <span>Sorted by |activation|</span>
      </div>
    </div>
  );
}
