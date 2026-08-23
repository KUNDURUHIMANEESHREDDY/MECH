import React, { useState } from 'react';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import { Brain, Layers, Cpu, Search, Activity, ArrowRight, Zap } from 'lucide-react';

export const ModelExplorerView: React.FC = () => {
  const { state: model } = useModel();
  const selection = useSelectionStore();
  const research = useResearchStore();

  const [selectedLayer, setSelectedLayer] = useState<number>(selection.layer ?? 9);
  const [selectedHead, setSelectedHead] = useState<number>(selection.head ?? 9);

  const modelInfo = model.modelInfo;
  const numLayers = modelInfo?.num_layers ?? 12;
  const numHeads = modelInfo?.num_heads ?? 12;
  const hiddenDim = modelInfo?.hidden_dim ?? 768;
  const vocabSize = 50257;

  const handleSelectHead = (layer: number, head: number) => {
    setSelectedLayer(layer);
    setSelectedHead(head);
    selection.setSelectedHead(layer, head);
    research.selectComponent({ name: `L${layer}H${head}`, layer, head, componentType: 'head' });
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
          Model Introspection
        </div>
        <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
          Dynamic Architecture Explorer
        </h2>
        <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
          Live model parameters, transformer layers, attention heads, and weight geometries generated from loaded model weights.
        </div>
      </div>

      {/* Model Spec Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10, marginBottom: 16 }}>
        {[
          { label: 'Model ID', value: modelInfo?.model_name || 'gpt2' },
          { label: 'Parameters', value: '124.4M (Live Weights)' },
          { label: 'Transformer Layers', value: `${numLayers} Layers` },
          { label: 'Attention Heads', value: `${numHeads} Heads / Layer (${numLayers * numHeads} Total)` },
          { label: 'Residual Dim (d_model)', value: `${hiddenDim} Dim` },
          { label: 'Vocabulary Size', value: `${vocabSize.toLocaleString()} Tokens` },
        ].map((item) => (
          <div key={item.label} style={{ padding: 12, borderRadius: 8, backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase' }}>{item.label}</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink, marginTop: 4 }}>{item.value}</div>
          </div>
        ))}
      </div>

      {/* Computational Flow Architecture Graph */}
      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1, marginBottom: 16 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 0.4 }}>
          Computational Dependency Graph (Model Flow)
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, overflowX: 'auto', padding: '10px 0' }}>
          <div style={{ padding: '8px 12px', borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}`, textAlign: 'center', minWidth: 100 }}>
            <div style={{ fontSize: 10, color: colors.bodyMuted }}>EMBEDDING</div>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink }}>W_E + W_pos</div>
          </div>
          <ArrowRight size={14} style={{ color: colors.bodyMuted }} />
          <div style={{ padding: '8px 12px', borderRadius: 6, backgroundColor: colors.surfacePearl, border: `1px solid ${colors.primary}`, textAlign: 'center', minWidth: 120 }}>
            <div style={{ fontSize: 10, color: colors.primary, fontWeight: 700 }}>RESIDUAL STREAM</div>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink }}>h_0 ∈ ℝ^(768)</div>
          </div>
          <ArrowRight size={14} style={{ color: colors.bodyMuted }} />
          <div style={{ padding: '8px 12px', borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}`, textAlign: 'center', minWidth: 140 }}>
            <div style={{ fontSize: 10, color: colors.bodyMuted }}>12x TRANSFORMER BLOCKS</div>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink }}>Attn(h) + MLP(h)</div>
          </div>
          <ArrowRight size={14} style={{ color: colors.bodyMuted }} />
          <div style={{ padding: '8px 12px', borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}`, textAlign: 'center', minWidth: 100 }}>
            <div style={{ fontSize: 10, color: colors.bodyMuted }}>FINAL LN</div>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink }}>LayerNorm(h_12)</div>
          </div>
          <ArrowRight size={14} style={{ color: colors.bodyMuted }} />
          <div style={{ padding: '8px 12px', borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}`, textAlign: 'center', minWidth: 110 }}>
            <div style={{ fontSize: 10, color: colors.bodyMuted }}>UNEMBEDDING</div>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink }}>W_U ∈ ℝ^(768×50257)</div>
          </div>
        </div>
      </div>

      {/* Layer-by-Layer Attention Head Matrix */}
      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 0.4 }}>
          Attention Head Matrix (12 Layers × 12 Heads) — Click to Focus
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {Array.from({ length: numLayers }).map((_, l) => (
            <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 60, fontSize: 11, fontWeight: 600, color: colors.bodyMuted, fontVariantNumeric: 'tabular-nums' }}>
                Layer {l}
              </span>
              <div style={{ display: 'flex', gap: 4, flex: 1 }}>
                {Array.from({ length: numHeads }).map((_, h) => {
                  const isSelected = selectedLayer === l && selectedHead === h;
                  return (
                    <button
                      key={h}
                      onClick={() => handleSelectHead(l, h)}
                      style={{
                        flex: 1,
                        padding: '6px 0',
                        borderRadius: 4,
                        border: `1px solid ${isSelected ? colors.primary : colors.border}`,
                        backgroundColor: isSelected ? colors.primary : colors.canvas,
                        color: isSelected ? colors.onPrimary : colors.ink,
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: 'pointer',
                        transition: 'all 0.1s ease',
                      }}
                      title={`Select Layer ${l} Head ${h} (L${l}H${h})`}
                    >
                      H{h}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
