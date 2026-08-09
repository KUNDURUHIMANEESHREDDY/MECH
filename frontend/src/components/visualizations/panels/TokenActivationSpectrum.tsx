import React from 'react';
import { colors } from '../../../design/tokens/colors';

/**
 * TokenActivationSpectrum — token-by-token activation bars for a single
 * selected neuron. Restored minimal implementation (the legacy file had
 * been gutted to a deprecated stub).
 */
interface Props {
  tokens?: string[];
  activations?: number[] | null;
}

const MAX_BAR_H = 64;
const BAR_W = 18;
const GAP = 4;

export function TokenActivationSpectrum({ tokens, activations }: Props) {
  if (!tokens || !activations) return null;
  const rows = Math.min(tokens.length, activations.length);
  if (rows === 0) return null;

  const maxVal = Math.max(...activations, 0.01);

  return (
    <div style={{ display: 'flex', gap: GAP, alignItems: 'flex-end', flexWrap: 'wrap', padding: '4px 2px' }}>
      {tokens.slice(0, rows).map((tok, i) => {
        const v = activations[i] ?? 0;
        const h = Math.max(2, Math.round((v / maxVal) * MAX_BAR_H));
        return (
          <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <div
              style={{
                width: BAR_W,
                height: h,
                background: v > 0.5 * maxVal ? colors.primary : colors.accentSoft,
                borderRadius: 2,
              }}
            />
            <span style={{ fontSize: 9, color: colors.bodyMuted, maxWidth: BAR_W + 8, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {tok}
            </span>
          </div>
        );
      })}
    </div>
  );
}
