import React, { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Play, Zap, Brain, Layers, ArrowRight, Activity, Sparkles } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import { scienceApi } from '../../science/api/scienceApi';
import { FeatureEvidence } from '../../science/types/scientificTypes';
import type { FC, NeuronData, PanelContext } from '../../shared/types';

const card: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: 12,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
};
const btn: React.CSSProperties = {
  background: colors.primary,
  color: colors.onPrimary,
  border: 'none',
  borderRadius: 6,
  padding: '6px 14px',
  fontSize: 12,
  fontWeight: 600,
  cursor: 'pointer',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
};
const selectStyle: React.CSSProperties = {
  padding: '5px 8px',
  borderRadius: 6,
  border: `1px solid ${colors.hairline}`,
  background: colors.canvas,
  fontSize: 12,
  color: colors.ink,
};

/** Token-firing intensity color for a neuron's per-token activation. */
function intensity(t: number, min: number, max: number): string {
  const k = max > min ? (t - min) / (max - min) : 1;
  return `rgba(124, 58, 237, ${(0.08 + 0.55 * k).toFixed(3)})`;
}

export const Gpt2NeuronExplorerPanel: FC<PanelContext> = () => {
  const { state: model, listModels, load, infer, clearError } = useModel();
  const sel = useSelectionStore();

  const [prompt, setPrompt] = useState('The capital of France is');
  const [layerNum, setLayerNum] = useState(0);
  const [activeTab, setActiveTab] = useState<'substrate' | 'sae'>('substrate');
  const [saeFeatures, setSaeFeatures] = useState<FeatureEvidence[]>([]);
  const [selectedFeatureId, setSelectedFeatureId] = useState<string | null>(null);

  useEffect(() => {
    if (model.availableModels.length === 0 && !model.loading) void listModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    scienceApi.fetchLayerFeatures(layerNum, model.modelInfo?.model_name || 'gpt2')
      .then((feats) => {
        setSaeFeatures(feats);
        if (feats.length > 0 && !selectedFeatureId) {
          setSelectedFeatureId(feats[0].feature_id);
        }
      })
      .catch((err) => console.warn('Could not load layer SAE features:', err));
  }, [layerNum, model.modelInfo?.model_name]);

  const layers = model.result?.layers ?? [];
  const tokens = model.result?.tokens.map((t) => t.text) ?? [];
  const layer = layerNum < layers.length ? layers[layerNum] : layers[0];

  const layerNeurons = useMemo((): NeuronData[] => {
    if (!layer) return [];
    const map = new Map<number, NeuronData>();
    for (const h of layer.heads) {
      for (const n of h.neurons) {
        const prev = map.get(n.index);
        if (!prev || n.activation > prev.activation) {
          map.set(n.index, { index: n.index, activation: n.activation, tokenActivations: n.tokenActivations });
        }
      }
    }
    return Array.from(map.values()).sort((a, b) => b.activation - a.activation);
  }, [layer]);

  const topNeurons = layerNeurons.slice(0, 40);
  const maxActivation = topNeurons[0]?.activation ?? 1;
  const selected = useMemo(
    () => topNeurons.find((n) => n.index === sel.neuron) ?? topNeurons[0] ?? null,
    [topNeurons, sel.neuron],
  );

  const strip = useMemo(() => {
    if (!selected?.tokenActivations) return null;
    const vals = selected.tokenActivations;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    return { vals, min, max };
  }, [selected]);

  const selectedFeature = saeFeatures.find((f) => f.feature_id === selectedFeatureId) || saeFeatures[0];

  const handleRun = async () => {
    clearError();
    await infer(prompt);
  };

  const handleSelect = (neuron: NeuronData) => {
    if (layer) sel.setSelectedNeuron(layer.index, neuron.index);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>
      {!model.result && (
        <>
          <div style={card}>
            <div style={{ fontWeight: 700, color: colors.ink }}>Model Selection</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
              <select value={model.availableModels[0] ?? 'gpt2'} onChange={(e) => load(e.target.value)} style={selectStyle}>
                {(model.availableModels.length ? model.availableModels : ['gpt2']).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <button onClick={() => load(model.availableModels[0] ?? 'gpt2')} disabled={model.loading} style={{ ...btn, opacity: model.loading ? 0.5 : 1 }}>
                {model.loading ? <span>Loading…</span> : <><Play size={13} /> Load Model</>}
              </button>
            </div>
          </div>
          <div style={card}>
            <div style={{ fontWeight: 700, color: colors.ink }}>Probe Sequence</div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={2}
              style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, background: colors.canvas, color: colors.ink }}
            />
            <div>
              <button onClick={handleRun} disabled={!model.loaded || model.running} style={{ ...btn, opacity: !model.loaded || model.running ? 0.5 : 1 }}>
                {model.running ? <span>Running…</span> : <><Play size={13} /> Run Probe</>}
              </button>
            </div>
          </div>
        </>
      )}

      {model.error && (
        <div style={{ background: colors.dangerSoft, color: colors.dangerText, border: `1px solid ${colors.dangerBorder}`, borderRadius: 8, padding: '8px 12px', display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 }}>
          <AlertTriangle size={14} /> {model.error}
        </div>
      )}

      {layer && (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: colors.bodyMuted }}>
                <Zap size={13} color={colors.purple} /> Layer
                <select value={layerNum < layers.length ? layerNum : 0} onChange={(e) => setLayerNum(Number(e.target.value))} style={selectStyle}>
                  {layers.map((l) => (
                    <option key={l.index} value={l.index}>L{l.index}</option>
                  ))}
                </select>
              </label>
              <span style={{ fontSize: 11, color: colors.bodyMuted }}>
                Top {topNeurons.length} of {layerNeurons.length} active substrate neurons
              </span>
            </div>

            {/* Substrate vs SAE Switcher */}
            <div style={{ display: 'flex', gap: 4, background: 'rgba(0,0,0,0.1)', padding: 2, borderRadius: 6 }}>
              <button
                onClick={() => setActiveTab('substrate')}
                style={{
                  padding: '4px 10px',
                  fontSize: 11,
                  fontWeight: 600,
                  border: 'none',
                  borderRadius: 4,
                  background: activeTab === 'substrate' ? colors.primary : 'transparent',
                  color: activeTab === 'substrate' ? '#fff' : colors.bodyMuted,
                  cursor: 'pointer',
                }}
              >
                Physical Substrate
              </button>
              <button
                onClick={() => setActiveTab('sae')}
                style={{
                  padding: '4px 10px',
                  fontSize: 11,
                  fontWeight: 600,
                  border: 'none',
                  borderRadius: 4,
                  background: activeTab === 'sae' ? '#10b981' : 'transparent',
                  color: activeTab === 'sae' ? '#fff' : colors.bodyMuted,
                  cursor: 'pointer',
                }}
              >
                SAE Dictionary ({saeFeatures.length})
              </button>
            </div>
          </div>

          {activeTab === 'substrate' ? (
            <>
              {/* Scientific Note */}
              <div style={{ padding: '8px 12px', background: 'rgba(124, 58, 237, 0.08)', border: '1px solid rgba(124, 58, 237, 0.2)', borderRadius: 6, fontSize: 11 }}>
                <strong>Reference Substrate Anchor: </strong>
                Physical neurons (L{layer.index}_N*) are dense, polysemantic substrate coordinates rather than isolated monosemantic concepts.
              </div>

              <div style={card}>
                <div style={{ fontWeight: 700, color: colors.ink }}>
                  Physical Substrate Activations · Layer {layer.index}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {topNeurons.map((n, i) => {
                    const active = selected?.index === n.index;
                    return (
                      <div
                        key={n.index}
                        onClick={() => handleSelect(n)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                          cursor: 'pointer',
                          padding: '4px 8px',
                          borderRadius: 5,
                          background: active ? colors.purpleSoft : 'transparent',
                        }}
                      >
                        <span style={{ width: 68, fontFamily: 'monospace', fontSize: 11, color: active ? colors.ink : colors.bodyMuted }}>
                          L{layer.index}_N{n.index}
                        </span>
                        <div style={{ flex: 1, height: 10, borderRadius: 5, background: colors.hairline, overflow: 'hidden' }}>
                          <div
                            style={{
                              height: '100%',
                              width: `${Math.max(2, (n.activation / maxActivation) * 100).toFixed(1)}%`,
                              background: active ? colors.primary : `rgba(124, 58, 237, ${(0.15 + 0.5 * (n.activation / maxActivation)).toFixed(3)})`,
                              borderRadius: 5,
                            }}
                          />
                        </div>
                        <span style={{ width: 56, textAlign: 'right', fontFamily: 'monospace', fontSize: 11, color: colors.bodyMuted }}>
                          {n.activation.toFixed(3)}
                        </span>
                        <span style={{ width: 20, textAlign: 'center' }}>{i === 0 ? '★' : ''}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {selected && (
                <div style={card}>
                  <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', justifyContent: 'space-between' }}>
                    <span>Firing Pattern · Substrate Anchor L{layer.index}_N{selected.index}</span>
                    <span style={{ fontSize: 11, color: colors.bodyMuted }}>Polysemantic Coordinate</span>
                  </div>
                  {strip ? (
                    <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                      {strip.vals.map((v, ti) => (
                        <span
                          key={ti}
                          title={`${tokens[ti] ?? ti}: ${v.toFixed(3)}`}
                          style={{
                            padding: '3px 6px',
                            borderRadius: 4,
                            fontSize: 11,
                            fontFamily: 'monospace',
                            background: intensity(v, strip.min, strip.max),
                            color: colors.ink,
                          }}
                        >
                          {tokens[ti] ?? ti}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: 12, color: colors.bodyMuted }}>No per-token activations recorded for this neuron.</div>
                  )}
                </div>
              )}
            </>
          ) : (
            /* SAE Dictionary View */
            <>
              <div style={{ padding: '8px 12px', background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: 6, fontSize: 11 }}>
                <strong>Candidate Semantic Units (SAE Dictionary): </strong>
                Sparse features scored on empirical Specificity, Consistency, and Cross-Prompt Stability.
              </div>

              <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
                {saeFeatures.map((feat) => (
                  <button
                    key={feat.feature_id}
                    onClick={() => setSelectedFeatureId(feat.feature_id)}
                    style={{
                      padding: '6px 10px',
                      fontSize: 11,
                      fontWeight: 600,
                      borderRadius: 4,
                      border: selectedFeature?.feature_id === feat.feature_id ? '1px solid #10b981' : `1px solid ${colors.hairline}`,
                      background: selectedFeature?.feature_id === feat.feature_id ? 'rgba(16, 185, 129, 0.15)' : colors.canvas,
                      color: selectedFeature?.feature_id === feat.feature_id ? '#10b981' : colors.bodyMuted,
                      cursor: 'pointer',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {feat.feature_id.split('_').slice(-1)[0]}
                  </button>
                ))}
              </div>

              {selectedFeature && (
                <div style={card}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 700, color: colors.primary, fontSize: 13 }}>
                      {selectedFeature.feature_id} (Layer {selectedFeature.layer})
                    </span>
                    <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 6px', borderRadius: 4, background: 'rgba(124, 58, 237, 0.15)', color: colors.primary }}>
                      {selectedFeature.evidence_level}
                    </span>
                  </div>

                  <div style={{ fontSize: 12, fontWeight: 600, color: colors.ink }}>
                    {selectedFeature.semantic_label}
                  </div>

                  {/* Empirical Quality Scores */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11, color: colors.bodyMuted }}>
                    <div>Uncertainty: <strong style={{ color: colors.ink }}>{selectedFeature.uncertainty !== undefined ? (selectedFeature.uncertainty * 100).toFixed(0) + '%' : 'N/A'}</strong></div>
                    <div>Evidence Level: <strong style={{ color: colors.ink }}>{selectedFeature.evidence_level || 'N/A'}</strong></div>
                    <div>Sample Size: <strong style={{ color: colors.ink }}>{selectedFeature.sample_size || 'N/A'}</strong></div>
                    <div>Method: <strong style={{ color: '#10b981' }}>{selectedFeature.methodology || 'N/A'}</strong></div>
                  </div>

                  {/* Feature -> Logit Linear Projection (W_U d_i) */}
                  <div style={{ padding: 8, background: 'rgba(0,0,0,0.03)', borderRadius: 4, border: `1px solid ${colors.hairline}` }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: colors.ink, marginBottom: 4 }}>
                      Linear Feature-to-Logit Projection (\(W_U \mathbf{d}_i\)):
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 4, fontSize: 11 }}>
                      {Object.entries(selectedFeature.linear_logit_delta || {}).map(([tok, delta]) => (
                        <div key={tok} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 6px', background: colors.canvas, borderRadius: 3, border: `1px solid ${colors.hairline}` }}>
                          <span>'{tok}'</span>
                          <strong style={{ color: delta > 0 ? '#10b981' : '#ef4444' }}>
                            {delta > 0 ? `+${delta.toFixed(2)}` : delta.toFixed(2)}
                          </strong>
                        </div>
                      ))}
                    </div>
                    <div style={{ fontSize: 10, color: colors.bodyMuted, marginTop: 6, fontStyle: 'italic' }}>
                      ⚠️ Linear projection behavior (\(\Delta z_i \approx a_i \cdot W_U \mathbf{d}_i\)) — reflects unembedding geometry, not causal evidence.
                    </div>
                  </div>

                  {/* Substrate Coordinates */}
                  {selectedFeature.substrate_anchors && (
                    <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                      Substrate Anchor Coordinates: {selectedFeature.substrate_anchors.map((a) => `L${a.layer}_N${a.neuron_idx}`).join(', ')}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
};