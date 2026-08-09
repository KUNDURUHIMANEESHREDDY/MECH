import React, { useMemo, useState } from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import { colors } from '../../design/tokens/colors';

/**
 * Sparse feature view over the real MLP activations recorded in the model
 * result. For the selected layer, every neuron with non-zero activation is
 * treated as an active dictionary feature; cells are sized/colored by
 * activation magnitude and the top-feeding tokens are shown on selection.
 */
const SAEFeatureBody: FC<PanelContext> = () => {
  const { state: model } = useModel();
  const sel = useSelectionStore();
  const [layerIdx, setLayerIdx] = useState<number | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<number | null>(null);

  const tokens = model.result?.tokens.map(t => t.text) ?? [];
  const layers = model.result?.layers ?? [];
  const selectedLayer = layerIdx ?? sel.layer ?? (layers.length > 0 ? 0 : null);
  const layer = selectedLayer !== null ? layers[selectedLayer] : undefined;

  const atoms = useMemo(() => {
    if (!layer) return [] as Array<{ index: number; activation: number; tokenIdx: number; strength: number }>;
    const atoms: Array<{ index: number; activation: number; tokenIdx: number; strength: number }> = [];
    // Neurons are attached to every head of the layer (activation grid is per layer).
    const neurons = layer.heads.flatMap(h => h.neurons);
    for (const n of neurons) {
      if ((n.activation ?? 0) <= 1e-6) continue;
      let tokenIdx = -1;
      let strength = 1;
      if (n.tokenActivations && n.tokenActivations.length > 0) {
        let max = -Infinity;
        n.tokenActivations.forEach((a, i) => {
          if (a > max) { max = a; tokenIdx = i; }
        });
        strength = max;
      }
      atoms.push({ index: n.index, activation: n.activation, tokenIdx, strength });
    }
    atoms.sort((a, b) => b.activation - a.activation);
    return atoms.slice(0, 80);
  }, [layer]);

  const maxActivation = atoms.length > 0 ? atoms[0].activation : 1;
  const selected = atoms.find(a => a.index === selectedFeature) ?? atoms[0];

  return (
    <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 12, overflow: 'auto', height: '100%', backgroundColor: colors.canvas }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4 }}>
          SAE features — sparse activations
        </div>
        {layers.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: colors.bodyMuted }}>
            <span>Layer</span>
            <select
              value={selectedLayer ?? 0}
              onChange={e => setLayerIdx(parseInt(e.target.value, 10))}
              style={{
                padding: '5px 8px', borderRadius: 6, border: `1px solid ${colors.border}`,
                backgroundColor: colors.canvasParchment, color: colors.ink, fontSize: 12, outline: 'none',
              }}
            >
              {layers.map(l => <option key={l.index} value={l.index}>L{l.index}</option>)}
            </select>
          </div>
        )}
      </div>

      {atoms.length === 0 ? (
        <div style={{ fontSize: 13, color: colors.bodyMuted, lineHeight: 1.5 }}>
          No features active yet. Run an inference to decode real sparse
          feature activations (non-zero neurons) for each layer.
        </div>
      ) : (
        <>
          {/* Feature grid */}
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 8, padding: 12, backgroundColor: colors.surfaceTile1 }}>
            <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.3 }}>
              {atoms.length} active features, {maxActivation.toFixed(3)} max
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {atoms.map((a) => {
                const active = selected && a.index === selected.index;
                const pct = (a.activation / maxActivation) * 100;
                return (
                  <button
                    key={a.index}
                    onClick={() => setSelectedFeature(a.index)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 8, width: '100%',
                      border: active ? `1px solid ${colors.primary}` : `1px solid ${colors.hairline}`,
                      borderRadius: 6, backgroundColor: active ? colors.accentSoft : colors.canvas,
                      padding: '4px 8px', cursor: 'pointer', textAlign: 'left',
                    }}
                  >
                    <span style={{ width: 64, fontSize: 11, color: colors.bodyMuted, fontVariantNumeric: 'tabular-nums' }}>
                      f{a.index}
                    </span>
                    <div style={{ flex: 1, height: 8, backgroundColor: colors.surfacePearl, borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${pct}%`, backgroundColor: a.activation > maxActivation * 0.6 ? colors.danger : colors.purple, borderRadius: 4 }} />
                    </div>
                    <span style={{ width: 44, fontSize: 11, color: colors.bodyMuted, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                      {a.activation.toFixed(2)}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Selected feature detail */}
          {selected && (
            <div style={{ border: `1px solid ${colors.border}`, borderRadius: 8, padding: 12, backgroundColor: colors.surfacePearl }}>
              <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.3 }}>
                Feature f{selected.index} · activation {selected.activation.toFixed(3)}
              </div>
              {selected.tokenIdx >= 0 ? (
                <div style={{ fontSize: 12, color: colors.ink, lineHeight: 1.5 }}>
                  Strongest trigger token: <strong>“{tokens[selected.tokenIdx] ?? `#${selected.tokenIdx}`}”</strong> (position {selected.tokenIdx})
                  <div style={{ marginTop: 8, fontSize: 11, color: colors.bodyMuted }}>
                    Token context:{' '}
                    <span style={{ color: colors.body }}>
                      {tokens.slice(Math.max(0, selected.tokenIdx - 2), Math.min(tokens.length, selected.tokenIdx + 3)).join(' ')}
                    </span>
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: 12, color: colors.bodyMuted }}>No per-token activation data for this feature.</div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};

pluginRegistry.register({
  id: 'sae_feature',
  title: 'SAE Feature Inspector',
  icon: 'Dna',
  category: 'sae',
  resourceKinds: ['sae', 'feature', 'model'],
  defaultDock: 'right',
  Body: SAEFeatureBody,
});