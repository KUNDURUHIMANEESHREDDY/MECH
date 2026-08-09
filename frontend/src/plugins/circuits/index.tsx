import React, { useMemo, useState } from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import { colors } from '../../design/tokens/colors';

interface HeadStats {
  layer: number;
  head: number;
  entropy: number;
  concentration: number;
  prevToken: number;
  earliestToken: number;
}

/**
 * Real circuit metrics computed from the actual attention matrices of the
 * loaded model result:
 * - entropy: how spread each head's attention is (low entropy = focused head)
 * - prevToken: average attention paid to the immediately previous token
 *   (induction-like behavior)
 * - earliestToken: average attention to the first token (priming / prefix)
 */
function computeHeadStats(
  layers: Array<{
    index: number;
    heads: Array<{ index: number; attentionMatrix: number[][] }>;
  }>,
): HeadStats[] {
  const stats: HeadStats[] = [];
  for (const layer of layers) {
    for (const head of layer.heads) {
      const m = head.attentionMatrix;
      if (!m || m.length === 0) continue;
      const n = m.length;
      let entropy = 0;
      let prevToken = 0;
      let earliestToken = 0;
      for (let i = 0; i < n; i++) {
        const row = m[i];
        let rowSum = row.reduce((a, b) => a + b, 0);
        if (rowSum <= 1e-9) rowSum = 1;
        for (const v of row) {
          const p = v / rowSum;
          if (p > 0) entropy -= p * Math.log2(p);
        }
        if (i > 0) prevToken += row[i - 1] ?? 0;
        earliestToken += row[0] ?? 0;
      }
      stats.push({
        layer: layer.index,
        head: head.index,
        entropy: entropy / n,
        concentration: prevToken + earliestToken,
        prevToken: prevToken / n,
        earliestToken: earliestToken / n,
      });
    }
  }
  return stats;
}

const CircuitExplorerBody: FC<PanelContext> = () => {
  const { state: model } = useModel();
  const sel = useSelectionStore();
  const [minWeight, setMinWeight] = useState(0.12);

  const layers = model.result?.layers ?? [];

  const stats = useMemo(() => computeHeadStats(layers), [layers]);

  const maxConcentration = Math.max(...stats.map(s => s.concentration), 1e-9);
  const maxEntropy = Math.max(...stats.map(s => s.entropy), 1e-9);

  // Active head from workspace selection (fall back to first head)
  const activeLayer = sel.layer ?? 0;
  const activeHeadIdx = (sel.head ?? 0);
  const activeStats = stats.find(s => s.layer === activeLayer && s.head === activeHeadIdx);

  return (
    <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 12, overflow: 'auto', height: '100%', backgroundColor: colors.canvas }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4 }}>
        Circuit graph — attention aggregation
      </div>

      {layers.length === 0 ? (
        <div style={{ fontSize: 13, color: colors.bodyMuted, lineHeight: 1.5 }}>
          No activations loaded yet. Run an inference to compute real per-layer,
          per-head attention circuit metrics (focal heads, induction-like
          previous-token attention, prefix priming).
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: colors.bodyMuted }}>
            <span>Edge threshold</span>
            <input
              type="range"
              min={0}
              max={0.6}
              step={0.01}
              value={minWeight}
              onChange={e => setMinWeight(parseFloat(e.target.value))}
              style={{ flex: 1, accentColor: colors.primary }}
            />
            <span style={{ width: 36, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{minWeight.toFixed(2)}</span>
          </div>

          {/* Focused heads (low entropy, high concentration) */}
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 8, padding: 12, backgroundColor: colors.surfaceTile1 }}>
            <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.3 }}>
              Focal heads (low entropy, high concentration) — {stats.length} total
            </div>
            {stats.length === 0 ? (
              <div style={{ fontSize: 12, color: colors.bodyMuted }}>No attention data.</div>
            ) : (
              stats
                .slice()
                .sort((a, b) => b.concentration - a.concentration)
                .slice(0, 10)
                .map((s, i) => (
                  <div key={`${s.layer}-${s.head}`} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
                    <span style={{ width: 22, fontSize: 11, color: colors.bodyMuted, textAlign: 'right' }}>{i + 1}</span>
                    <span style={{ width: 92, fontSize: 12, color: colors.ink, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                      L{s.layer}·H{s.head}
                    </span>
                    <div style={{ flex: 1, height: 9, backgroundColor: colors.surfacePearl, borderRadius: 5, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${(s.concentration / maxConcentration) * 100}%`, backgroundColor: colors.purple, borderRadius: 5 }} />
                    </div>
                    <span style={{ width: 44, fontSize: 11, color: colors.bodyMuted, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                      {s.concentration.toFixed(2)}
                    </span>
                  </div>
                )))}
          </div>

          {/* Induction-like edges (previous-token attention) */}
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 8, padding: 12, backgroundColor: colors.surfaceTile1 }}>
            <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.3 }}>
              Induction-like edges (previous-token attention above {minWeight.toFixed(2)})
            </div>
            {stats.length === 0 ? (
              <div style={{ fontSize: 12, color: colors.bodyMuted }}>No attention data.</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {stats
                  .filter(s => s.prevToken >= minWeight)
                  .sort((a, b) => b.prevToken - a.prevToken)
                  .slice(0, 12)
                  .map((s, i) => (
                    <div key={`prev-${s.layer}-${s.head}`} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ width: 22, fontSize: 11, color: colors.bodyMuted, textAlign: 'right' }}>{i + 1}</span>
                      <span style={{ width: 92, fontSize: 12, color: colors.ink, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                        L{s.layer}·H{s.head}
                      </span>
                      <div style={{ flex: 1, height: 8, backgroundColor: colors.surfacePearl, borderRadius: 5, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${(s.prevToken / maxConcentration) * 100}%`, backgroundColor: colors.primary, borderRadius: 5 }} />
                      </div>
                      <span style={{ width: 44, fontSize: 11, color: colors.bodyMuted, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                        {s.prevToken.toFixed(2)}
                      </span>
                    </div>
                  ))}
              </div>
            )}
          </div>

          {/* Selected head detail */}
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 8, padding: 12, backgroundColor: colors.surfacePearl }}>
            <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.3 }}>
              Selected head L{activeLayer}·H{activeHeadIdx}
            </div>
            {activeStats ? (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                {[
                  { label: 'Entropy', value: activeStats.entropy, max: maxEntropy, color: colors.danger },
                  { label: 'Prev-token', value: activeStats.prevToken, max: maxConcentration, color: colors.primary },
                  { label: 'Prefix', value: activeStats.earliestToken, max: maxConcentration, color: colors.purple },
                ].map(stat => (
                  <div key={stat.label} style={{ padding: 8, borderRadius: 6, backgroundColor: colors.canvas }}>
                    <div style={{ fontSize: 10, color: colors.bodyMuted, textTransform: 'uppercase' }}>{stat.label}</div>
                    <div style={{ fontSize: 15, fontWeight: 700, color: colors.ink, fontVariantNumeric: 'tabular-nums' }}>{stat.value.toFixed(3)}</div>
                    <div style={{ height: 4, backgroundColor: colors.surfacePearl, borderRadius: 2, marginTop: 6, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${(stat.value / stat.max) * 100}%`, backgroundColor: stat.color, borderRadius: 2 }} />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ fontSize: 12, color: colors.bodyMuted }}>
                Select a layer/head to inspect its circuit metrics.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

pluginRegistry.register({
  id: 'circuit_explorer',
  title: 'Circuit Explorer',
  icon: 'Activity',
  category: 'circuits',
  resourceKinds: ['circuit', 'model'],
  defaultDock: 'center',
  Body: CircuitExplorerBody,
});