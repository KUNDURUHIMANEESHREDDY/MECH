import React, { useEffect, useMemo, useState } from 'react';
import { Layers, Loader2, Play, AlertTriangle, ArrowRight } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import type { FC, PanelContext } from '../../shared/types';

const card: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: 12,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
  flexShrink: 0,
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
const nodeChip: React.CSSProperties = {
  padding: '3px 6px',
  borderRadius: 4,
  background: colors.canvas,
  border: `1px solid ${colors.hairline}`,
  fontSize: 9,
  fontWeight: 600,
  color: colors.inkMuted48,
  textAlign: 'center',
  alignSelf: 'center',
  width: '100%',
  boxSizing: 'border-box',
};

function headStrength(matrix: number[][]): number {
  if (!matrix || matrix.length === 0) return 0;
  let sum = 0;
  let count = 0;
  for (const row of matrix) {
    for (const v of row) {
      sum += v;
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

export const TransformerNetworkPanel: FC<PanelContext> = () => {
  const { state: model, listModels, load, infer, clearError } = useModel();
  const sel = useSelectionStore();

  const [prompt, setPrompt] = useState('The capital of France is');
  const [hovered, setHovered] = useState<number | null>(null);
  const [headNum, setHeadNum] = useState(0);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [hoveredHead, setHoveredHead] = useState<{ layer: number; head: number } | null>(null);

  useEffect(() => {
    if (model.availableModels.length === 0 && !model.loading) void listModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const layers = model.result?.layers ?? [];
  const tokens = model.result?.tokens ?? [];

  const headMeans = useMemo(() => {
    return layers.map((l) =>
      l.heads.map((h) => headStrength(h.attentionMatrix)),
    );
  }, [layers]);

  const layerMlps = useMemo(() => {
    return layers.map((l) => {
      const map = new Map<number, { index: number; activation: number }>();
      for (const h of l.heads) {
        for (const n of h.neurons) {
          const prev = map.get(n.index);
          if (!prev || n.activation > prev.activation) {
            map.set(n.index, { index: n.index, activation: n.activation });
          }
        }
      }
      return Array.from(map.values());
    });
  }, [layers]);

  const handleRun = async () => {
    clearError();
    await infer(prompt);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>
      {!model.loaded && (
        <div style={card}>
          <div style={{ fontWeight: 700, color: colors.ink }}>Model</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <select
              value={model.availableModels[0] ?? 'gpt2'}
              onChange={(e) => load(e.target.value)}
              style={selectStyle}
            >
              {(model.availableModels.length ? model.availableModels : ['gpt2']).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <button onClick={() => load(model.availableModels[0] ?? 'gpt2')} disabled={model.loading} style={{ ...btn, opacity: model.loading ? 0.5 : 1 }}>
              {model.loading ? <Loader2 size={13} className="spin" /> : <Play size={13} />} Load Model
            </button>
          </div>
        </div>
      )}

      {model.error && (
        <div style={{ background: colors.dangerSoft, color: colors.dangerText, border: `1px solid ${colors.dangerBorder}`, borderRadius: 8, padding: '8px 12px', display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 }}>
          <AlertTriangle size={14} /> {model.error}
        </div>
      )}

      {!model.result && (
        <div style={card}>
          <div style={{ fontWeight: 700, color: colors.ink }}>Run inference to inspect internals</div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, background: colors.canvas, color: colors.ink }}
          />
          <div>
            <button onClick={handleRun} disabled={!model.loaded || model.running} style={{ ...btn, opacity: !model.loaded || model.running ? 0.5 : 1 }}>
              {model.running ? <Loader2 size={13} className="spin" /> : <Play size={13} />} Run
            </button>
          </div>
        </div>
      )}

      {model.result && layers.length > 0 && (
        <>
          <div style={{ overflowX: 'auto', paddingBottom: 10 }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'stretch', width: 'max-content', minHeight: 240 }}>
              {/* Input tokens */}
              <div style={card}>
                <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Layers size={14} color={colors.primary} /> Input
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, maxWidth: 150 }}>
                  {tokens.map((t, i) => (
                    <div
                      key={i}
                      onMouseEnter={() => setHovered(i)}
                      onMouseLeave={() => setHovered(null)}
                      style={{
                        padding: '3px 6px',
                        borderRadius: 4,
                        fontSize: 10,
                        background: hovered === i ? colors.primary : colors.canvas,
                        color: hovered === i ? colors.onPrimary : colors.ink,
                        border: `1px solid ${hovered === i ? colors.primary : colors.hairline}`,
                        cursor: 'default',
                      }}
                    >
                      {t.text}
                      <span style={{ opacity: 0.6, marginLeft: 4, fontSize: 8 }}>{t.id}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ alignSelf: 'center' }}><ArrowRight size={14} color={colors.inkMuted48} /></div>

              {/* Embedding */}
              <div style={card} title="Token + position embeddings → 768d residual stream">
                <div style={{ fontWeight: 700, color: colors.ink }}>Embed</div>
                <div style={{ ...nodeChip, width: 'auto', padding: '8px 10px' }}>768d</div>
              </div>

              <div style={{ alignSelf: 'center' }}><ArrowRight size={14} color={colors.inkMuted48} /></div>

              {/* Transformer blocks, left to right */}
              {layers.map((l, li) => {
                const maxMean = Math.max(...headMeans[li], 0.0001);
                const mlp = layerMlps[li];
                const maxAct = Math.max(...mlp.map((n) => n.activation), 0.0001);
                const isLayerSelected = selectedLayer === li;
                return (
                  <React.Fragment key={li}>
                    <div
                      style={{
                        ...card,
                        cursor: 'pointer',
                        border: isLayerSelected ? `2px solid ${colors.primary}` : card.border,
                        boxShadow: isLayerSelected ? `0 0 0 2px ${colors.purpleBorder}` : undefined,
                      }}
                      onClick={() => setSelectedLayer(isLayerSelected ? null : li)}
                    >
                      <div style={{ fontWeight: 700, color: colors.ink, textAlign: 'center' }}>L{li}</div>
                      <div style={{ ...nodeChip }}>LN₁</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, justifyContent: 'center', maxWidth: 120 }}>
                        {l.heads.map((h) => {
                          const strength = headMeans[li][h.index] / maxMean;
                          const selected = sel.head === h.index && sel.layer === li;
                          const active = headNum === h.index && sel.layer === li;
                          const isHovered = hoveredHead?.layer === li && hoveredHead?.head === h.index;
                          return (
                            <div
                              key={h.index}
                              onClick={(e) => {
                                e.stopPropagation();
                                setHeadNum(h.index);
                                sel.setSelectedHead(li, h.index);
                              }}
                              onMouseEnter={() => setHoveredHead({ layer: li, head: h.index })}
                              onMouseLeave={() => setHoveredHead(null)}
                              title={`H${h.index} · mean attn ${headMeans[li][h.index].toFixed(3)}`}
                              style={{
                                width: 20,
                                height: 14,
                                borderRadius: 3,
                                background: selected ? colors.primary : active ? colors.purpleBorder : isHovered ? colors.surfacePearl : colors.canvas,
                                border: `1px solid ${selected ? colors.onDark : isHovered ? colors.primary : colors.hairline}`,
                                cursor: 'pointer',
                                opacity: selected ? 1 : 0.35 + strength * 0.65,
                                boxShadow: selected ? `0 0 0 2px ${colors.purpleBorder}` : undefined,
                                transform: isHovered ? 'scale(1.2)' : undefined,
                                transition: 'transform 0.1s ease',
                              }}
                            />
                          );
                        })}
                      </div>
                      <div style={{ ...nodeChip }}>LN₂</div>
                      <div style={{ display: 'flex', gap: 2, height: 34, alignItems: 'flex-end', justifyContent: 'center', maxWidth: 120 }}>
                        {mlp.slice(0, 48).map((n) => (
                          <div
                            key={n.index}
                            onClick={(e) => {
                              e.stopPropagation();
                              sel.setSelectedNeuron(li, n.index);
                            }}
                            title={`Neuron ${n.index} · act ${n.activation.toFixed(3)}`}
                            style={{
                              flex: 1,
                              minWidth: 2,
                              height: Math.max(3, (n.activation / maxAct) * 30),
                              background: colors.warning,
                              borderRadius: 1,
                              cursor: 'pointer',
                              opacity: 0.55 + (n.activation / maxAct) * 0.45,
                            }}
                          />
                        ))}
                      </div>
                      <div style={{ fontSize: 8, color: colors.bodyMuted, textAlign: 'center' }}>
                        {mlp.length} neurons
                      </div>
                    </div>
                    <div style={{ alignSelf: 'center' }}><ArrowRight size={14} color={colors.inkMuted48} /></div>
                  </React.Fragment>
                );
              })}

              {/* Output */}
              <div style={card}>
                <div style={{ fontWeight: 700, color: colors.ink }}>LM Head</div>
                <div style={{ ...nodeChip, width: 'auto', padding: '8px 10px' }}>50257 logits</div>
                <div style={{ fontSize: 9, color: colors.bodyMuted, maxWidth: 110, textAlign: 'center' }}>
                  {model.result.generatedText ? `→ ${model.result.generatedText.split(/\s+/).slice(-4).join(' ')}…` : 'next-token distribution'}
                </div>
              </div>
            </div>
          </div>

          <div style={{ fontSize: 11, color: colors.bodyMuted }}>
            GPU {model.result.gpuUtil != null ? `${model.result.gpuUtil}%` : '—'} · Mem {model.result.memoryUtil != null ? `${model.result.memoryUtil}%` : '—'} · {tokens.length} tokens · {layers.length} layers
          </div>

          {/* Layer Detail Panel */}
          {selectedLayer !== null && layers[selectedLayer] && (
            <div style={{ ...card, marginTop: 8 }}>
              <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Layers size={14} color={colors.primary} />
                Layer {selectedLayer} Details
              </div>
              
              {/* Attention Patterns */}
              <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 4 }}>Attention Patterns</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {layers[selectedLayer].heads.map((h) => {
                  const matrix = h.attentionMatrix;
                  const maxVal = Math.max(...matrix.flat(), 0.0001);
                  return (
                    <div key={h.index} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                      <div style={{ fontSize: 10, color: colors.ink, fontWeight: 600 }}>Head {h.index}</div>
                      <div
                        style={{
                          display: 'grid',
                          gridTemplateColumns: `repeat(${tokens.length}, 12px)`,
                          gap: 1,
                        }}
                      >
                        {matrix.map((row, ri) =>
                          row.map((val, ci) => (
                            <div
                              key={`${ri}-${ci}`}
                              title={`${tokens[ri]?.text ?? ri} → ${tokens[ci]?.text ?? ci}: ${val.toFixed(3)}`}
                              style={{
                                width: 12,
                                height: 12,
                                borderRadius: 2,
                                background: `rgba(99, 102, 241, ${val / maxVal})`,
                                cursor: 'pointer',
                              }}
                            />
                          ))
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Token List */}
              <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 8 }}>Tokens</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {tokens.map((t, i) => (
                  <div
                    key={i}
                    style={{
                      padding: '2px 6px',
                      borderRadius: 3,
                      fontSize: 10,
                      background: colors.surfacePearl,
                      color: colors.ink,
                      border: `1px solid ${colors.hairline}`,
                    }}
                  >
                    {t.text}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};