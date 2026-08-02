import React, { useState, useMemo } from 'react';
import { AttentionHeatmap } from './panels/AttentionHeatmap';
import { ActivationHeatmap } from './panels/ActivationHeatmap';
import { NeuronUMAP } from './neuron-umap/NeuronUMAP';
import { buildNeuronPoints, idForHeadNeuron, parseNeuronId } from './neuron-umap/data';

interface Props {
  tokens: string[];
  layers: any[];
  numLayers: number;
  numHeads: number;
  darkMode: boolean;
}

export function TransformerExplorer({ tokens, layers, numLayers, numHeads, darkMode }: Props) {
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedNeuron, setSelectedNeuron] = useState<number | null>(null);
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<'attention' | 'neurons' | 'spectrum' | 'compare' | 'umap'>('attention');
  const [compareHeads, setCompareHeads] = useState<number[]>([0, 1]);

  const layer = layers[selectedLayer];
  const head = layer?.heads[selectedHead];
  const neurons = head?.neurons ?? [];
  const matrix = head?.attentionMatrix ?? [];

  const headLabels = useMemo(() => Array.from({ length: numHeads }, (_, i) => `H${i}`), [numHeads]);

  const selNeuronData = selectedNeuron !== null ? neurons[selectedNeuron] : null;

  const umapPoints = useMemo(() => buildNeuronPoints(layers, tokens), [layers, tokens]);

  // Multi-head summary: compute average attention per head for the selected layer
  const layerHeadSummaries = useMemo(() => {
    if (!layer) return [];
    return layer.heads.map((h: any, i: number) => ({
      headIndex: i,
      avgActivation: h.attentionMatrix?.flat().reduce((a: number, b: number) => a + b, 0) / (h.attentionMatrix?.flat().length || 1),
      maxActivation: Math.max(...(h.attentionMatrix?.flat() ?? [0])),
    }));
  }, [layer]);

  // Token-level neuron activation: for each token, which neurons are most active
  const tokenNeuronMap = useMemo(() => {
    if (!selNeuronData?.tokenActivations || tokens.length === 0) return [];
    return tokens.map((t, i) => ({
      token: t,
      activation: selNeuronData.tokenActivations[i] ?? 0,
      idx: i,
    })).sort((a, b) => Math.abs(b.activation) - Math.abs(a.activation));
  }, [selNeuronData, tokens]);

  const bg = darkMode ? '#0f172a' : '#f8fafc';
  const cardBg = darkMode ? '#1e293b' : '#ffffff';
  const border = darkMode ? '#334155' : '#e2e8f0';
  const textColor = darkMode ? '#e2e8f0' : '#1e293b';
  const mutedColor = darkMode ? '#94a3b8' : '#64748b';

  const accent = '#3b82f6';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontFamily: 'system-ui, sans-serif', fontSize: 12, color: textColor }}>
      {/* ── Controls Bar ─────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', background: cardBg, padding: 10, borderRadius: 8, border: `1px solid ${border}` }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ color: mutedColor, fontWeight: 600 }}>Layer</span>
          <input type="range" min={0} max={numLayers - 1} value={selectedLayer}
            onChange={e => { setSelectedLayer(Number(e.target.value)); setSelectedHead(0); setSelectedNeuron(null); }}
            style={{ width: 100 }} />
          <span style={{ fontFamily: 'monospace', minWidth: 24, fontWeight: 700 }}>{selectedLayer}</span>
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ color: mutedColor, fontWeight: 600 }}>Head</span>
          <select value={selectedHead} onChange={e => { setSelectedHead(Number(e.target.value)); setSelectedNeuron(null); }}
            style={{ background: bg, color: textColor, border: `1px solid ${border}`, borderRadius: 4, padding: '2px 4px', fontSize: 11 }}>
            {headLabels.map((l, i) => <option key={i} value={i}>{l}</option>)}
          </select>
        </label>

        <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ color: mutedColor, fontWeight: 600 }}>Neuron</span>
          <input type="range" min={0} max={Math.max(neurons.length - 1, 0)} value={selectedNeuron ?? 0}
            onChange={e => setSelectedNeuron(Number(e.target.value))}
            style={{ width: 100 }} disabled={neurons.length === 0} />
          <span style={{ fontFamily: 'monospace', minWidth: 24, fontWeight: 700 }}>{selectedNeuron ?? '—'}</span>
        </label>

        <div style={{ display: 'flex', gap: 2, marginLeft: 'auto' }}>
          {(['attention', 'neurons', 'spectrum', 'compare', 'umap'] as const).map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              style={{
                background: activeTab === tab ? accent : bg,
                color: activeTab === tab ? '#fff' : textColor,
                border: `1px solid ${border}`, borderRadius: 4,
                padding: '3px 8px', cursor: 'pointer', fontSize: 11, fontWeight: 600,
              }}>
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* ── Multi-Head Overview (always visible in attention tab) ─── */}
      {activeTab === 'attention' && layer && (
        <div style={{ background: cardBg, borderRadius: 8, padding: 10, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 600, color: mutedColor, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6, fontSize: 11 }}>
            Layer {selectedLayer} — All Heads Overview
          </div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {layerHeadSummaries.map((s: any, i: number) => (
              <button key={i} onClick={() => { setSelectedHead(i); setSelectedNeuron(null); }}
                style={{
                  background: i === selectedHead ? accent : bg,
                  color: i === selectedHead ? '#fff' : textColor,
                  border: `1px solid ${i === selectedHead ? accent : border}`,
                  borderRadius: 4, padding: '4px 8px', cursor: 'pointer',
                  fontSize: 10, fontWeight: i === selectedHead ? 700 : 400,
                  minWidth: 36, textAlign: 'center',
                }}>
                H{i}
                <div style={{ fontSize: 9, opacity: 0.7 }}>{s.avgActivation.toFixed(2)}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* ── Main Content Grid ────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: activeTab === 'compare' ? '1fr 1fr' : '1fr 1fr', gap: 12 }}>
        {/* Attention Heatmap */}
        {(activeTab === 'attention' || activeTab === 'compare') && (
          <div style={{ background: cardBg, borderRadius: 8, padding: 12, border: `1px solid ${border}` }}>
            <div style={{ fontWeight: 600, color: mutedColor, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, fontSize: 11 }}>
              Attention Heatmap — L{selectedLayer} H{selectedHead}
            </div>
            {matrix.length > 0 && tokens.length > 0 ? (
              <AttentionHeatmap
                matrix={matrix}
                tokens={tokens}
                hoveredToken={hoveredToken}
                onHoverToken={setHoveredToken}
              />
            ) : (
              <div className="hint">Run a prompt first to populate attention data.</div>
            )}
            {hoveredToken !== null && tokens[hoveredToken] && (
              <div style={{ marginTop: 6, fontSize: 11, color: mutedColor }}>
                Hovering: <strong>"{tokens[hoveredToken]}"</strong> (token {hoveredToken})
              </div>
            )}
          </div>
        )}

        {/* Neuron Activation Panel */}
        {(activeTab === 'neurons' || activeTab === 'spectrum') && (
          <div style={{ background: cardBg, borderRadius: 8, padding: 12, border: `1px solid ${border}` }}>
            <div style={{ fontWeight: 600, color: mutedColor, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, fontSize: 11 }}>
              Neuron Activations — Layer {selectedLayer}
            </div>
            {neurons.length > 0 ? (
              <ActivationHeatmap
                activations={neurons.map((n: any) => n.activation)}
                neuronIndex={selectedNeuron}
                onSelectNeuron={(i: number) => setSelectedNeuron(i)}
                tokens={tokens}
                neuronTokenActivations={neurons.map((n: any) => n.tokenActivations ?? null)}
              />
            ) : (
              <div className="hint">No neuron data for this layer.</div>
            )}
          </div>
        )}

        {/* Compare Panel */}
        {activeTab === 'compare' && (
          <>
            <div style={{ background: cardBg, borderRadius: 8, padding: 12, border: `1px solid ${border}` }}>
              <div style={{ fontWeight: 600, color: mutedColor, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, fontSize: 11 }}>
                Compare Heads — Layer {selectedLayer}
              </div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <select value={compareHeads[0]} onChange={e => setCompareHeads([Number(e.target.value), compareHeads[1]])}
                  style={{ background: bg, color: textColor, border: `1px solid ${border}`, borderRadius: 4, padding: '2px 4px', fontSize: 11 }}>
                  {headLabels.map((l, i) => <option key={i} value={i}>{l}</option>)}
                </select>
                <span style={{ color: mutedColor, alignSelf: 'center' }}>vs</span>
                <select value={compareHeads[1]} onChange={e => setCompareHeads([compareHeads[0], Number(e.target.value)])}
                  style={{ background: bg, color: textColor, border: `1px solid ${border}`, borderRadius: 4, padding: '2px 4px', fontSize: 11 }}>
                  {headLabels.map((l, i) => <option key={i} value={i}>{l}</option>)}
                </select>
              </div>
              {layer?.heads[compareHeads[0]]?.attentionMatrix && layer?.heads[compareHeads[1]]?.attentionMatrix ? (
                <div style={{ display: 'flex', gap: 8 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 10, color: mutedColor, marginBottom: 4 }}>Head {compareHeads[0]}</div>
                    <AttentionHeatmap
                      matrix={layer.heads[compareHeads[0]].attentionMatrix}
                      tokens={tokens}
                      hoveredToken={null}
                      onHoverToken={() => {}}
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 10, color: mutedColor, marginBottom: 4 }}>Head {compareHeads[1]}</div>
                    <AttentionHeatmap
                      matrix={layer.heads[compareHeads[1]].attentionMatrix}
                      tokens={tokens}
                      hoveredToken={null}
                      onHoverToken={() => {}}
                    />
                  </div>
                </div>
              ) : (
                <div className="hint">Run a prompt first.</div>
              )}
            </div>

            <div style={{ background: cardBg, borderRadius: 8, padding: 12, border: `1px solid ${border}` }}>
              <div style={{ fontWeight: 600, color: mutedColor, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, fontSize: 11 }}>
                Head Similarity — Layer {selectedLayer}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {layerHeadSummaries.map((s: any, i: number) => {
                  const maxAvg = Math.max(...layerHeadSummaries.map((x: any) => x.avgActivation), 0.01);
                  const pct = (s.avgActivation / maxAvg) * 100;
                  return (
                    <div key={i} style={{
                      width: 36, height: 36, borderRadius: 4,
                      background: `rgba(59, 130, 246, ${0.1 + (pct / 100) * 0.9})`,
                      border: `1px solid ${border}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 9, fontWeight: 600, cursor: 'pointer',
                      color: pct > 50 ? '#fff' : textColor,
                    }}
                      onClick={() => { setSelectedHead(i); setActiveTab('attention'); }}
                      title={`H${i}: avg=${s.avgActivation.toFixed(3)}, max=${s.maxActivation.toFixed(3)}`}>
                      H{i}
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </div>

      {/* ── Neuron UMAP ─────────────────────────────────────── */}
      {activeTab === 'umap' && (
        <div style={{ background: cardBg, borderRadius: 8, padding: 12, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 600, color: mutedColor, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, fontSize: 11 }}>
            Neuron Map — All Layers & Heads
          </div>
          <NeuronUMAP
            points={umapPoints}
            tokens={tokens}
            selectedId={selectedNeuron !== null ? idForHeadNeuron(selectedLayer, selectedHead, selectedNeuron) : null}
            onSelectNeuron={id => {
              const parsed = parseNeuronId(id);
              if (!parsed) {
                setSelectedNeuron(null);
                return;
              }
              if (parsed.head !== undefined) {
                setSelectedLayer(parsed.layer);
                setSelectedHead(parsed.head);
              }
              setSelectedNeuron(parsed.neuron);
            }}
            darkMode={darkMode}
            height={460}
          />
        </div>
      )}

      {/* ── Token Detail: which neurons fire for this token ──── */}
      {hoveredToken !== null && tokens[hoveredToken] && neurons.length > 0 && (
        <div style={{ background: cardBg, borderRadius: 8, padding: 12, border: `1px solid ${border}` }}>
          <div style={{ fontWeight: 600, color: mutedColor, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, fontSize: 11 }}>
            Neurons Active for "{tokens[hoveredToken]}" (token {hoveredToken})
          </div>
          {neurons.length > 0 && (
            <ActivationHeatmap
              activations={neurons.map((n: any) => n.activation)}
              neuronIndex={selectedNeuron}
              onSelectNeuron={(i: number) => setSelectedNeuron(i)}
              tokens={tokens}
              neuronTokenActivations={neurons.map((n: any) => n.tokenActivations ?? null)}
            />
          )}
        </div>
      )}

      {/* ── Summary Bar ──────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11, color: mutedColor }}>
        <span>Layers: {numLayers}</span>
        <span>Heads/Layer: {numHeads}</span>
        <span>Neurons/Layer: {neurons.length}</span>
        <span>Tokens: {tokens.length}</span>
        {selNeuronData && (
          <span>Neuron #{selNeuronData.index} activation: {selNeuronData.activation.toFixed(4)}</span>
        )}
        {hoveredToken !== null && tokens[hoveredToken] && (
          <span>Token: "{tokens[hoveredToken]}"</span>
        )}
      </div>
    </div>
  );
}
