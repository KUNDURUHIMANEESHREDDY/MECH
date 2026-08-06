import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { api } from '../services/api';
import { colors } from '../design/tokens';
import { NeuronUMAP } from './visualizations/neuron-umap/NeuronUMAP';
import { buildLayerNeuronPoints, idForLayerNeuron, parseNeuronId } from './visualizations/neuron-umap/data';

/* ---------- sub-components ---------- */

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rgba(hex: string, a: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

function StatusBadge({ value }: { value: string }) {
  const ok = value === 'ok' || value === 'loaded';
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 4,
      fontSize: 11, fontWeight: 600, letterSpacing: '0.04em',
      background: ok ? colors.successSoft : colors.dangerSoft,
      color: ok ? colors.successText : colors.dangerText,
    }}>
      {value || '—'}
    </span>
  );
}

function Spinner() {
  return (
    <span style={{ marginLeft: 6, opacity: 0.7, display: 'inline-flex' }}>
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
        <path d="M2 1h8M2 11h8M2 1l4 5 4-5M2 11l4-5 4 5" />
      </svg>
    </span>
  );
}

function KV({ k, v, mono }: { k: string; v: string | number; mono?: boolean }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', marginBottom: 3, fontSize: 12 }}>
      <span style={{ color: colors.inkMuted48, minWidth: 150, flexShrink: 0 }}>{k}</span>
      <span style={mono ? { fontFamily: 'monospace' } : {}}>{v ?? '—'}</span>
    </div>
  );
}

/* ---------- Attention Heatmap ---------- */

function AttentionHeatmap({ matrix, tokens }: { matrix: number[][]; tokens: string[] }) {
  if (!matrix || !matrix.length) return <p className="hint" style={{ fontSize: 11 }}>No attention data.</p>;
  const max = Math.max(...matrix.flat(), 0.01);
  const cell = (v: number) => {
    const t = max > 0 ? Math.max(0, Math.min(1, v / max)) : 0;
    const stops = [hexToRgb(colors.surfaceBlack), hexToRgb(colors.primaryOnDark), hexToRgb(colors.warning)];
    const scaled = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(scaled));
    const f = scaled - i;
    const [r, g, b] = stops[i].map((ch, k) => Math.round(ch + (stops[i + 1][k] - ch) * f));
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    return { background: `rgb(${r},${g},${b})`, color: lum > 140 ? colors.ink : colors.onDark };
  };
  const sz = 28;
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: 9 }}>
        <thead>
          <tr>
            <th style={{ width: sz }} />
            {tokens.map((t, i) => (
              <th key={i} style={{ width: sz, textAlign: 'center', fontWeight: 500, padding: '1px 0',
                maxWidth: sz, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                title={t}>{t.slice(0, 3)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, qi) => (
            <tr key={qi}>
              <td style={{ fontSize: 9, fontWeight: 500, paddingRight: 2, whiteSpace: 'nowrap',
                maxWidth: 40, overflow: 'hidden', textOverflow: 'ellipsis' }} title={tokens[qi]}>
                {(tokens[qi] || '').slice(0, 4)}
              </td>
              {row.map((v, ki) => (
                <td key={ki} style={{
                  width: sz, height: sz, textAlign: 'center', ...cell(v),
                  border: `1px solid ${rgba(colors.bodyMuted, 0.12)}`, borderRadius: 1,
                  cursor: 'pointer', fontSize: 8,
                }} title={`Q=${tokens[qi]?.slice(0,4)} K=${tokens[ki]?.slice(0,4)} w=${v.toFixed(3)}`}>
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- Neuron bar chart ---------- */

function NeuronBarChart({ neurons, selectedNeuron, onSelect }: {
  neurons: Array<{ neuron_index: number; activation: number | null; in_weight_l2: number | null; out_weight_l2: number | null }>;
  selectedNeuron: number | null;
  onSelect: (i: number) => void;
}) {
  const maxAct = Math.max(...neurons.map(n => Math.abs(n.activation ?? 0)), 0.01);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      {neurons.map((n) => {
        const act = n.activation ?? 0;
        const pct = (Math.abs(act) / maxAct) * 100;
        const isSelected = selectedNeuron === n.neuron_index;
        const color = act >= 0
          ? rgba(colors.primary, 0.3 + 0.7 * pct / 100)
          : rgba(colors.danger, 0.3 + 0.7 * pct / 100);
        return (
          <div
            key={n.neuron_index}
            onClick={() => onSelect(n.neuron_index)}
            style={{
              display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer',
              padding: '2px 4px', borderRadius: 3,
              background: isSelected ? rgba(colors.primary, 0.15) : 'transparent',
              border: isSelected ? `1px solid ${colors.primary}` : '1px solid transparent',
              fontSize: 10, height: 18,
            }}
            title={`N${n.neuron_index} | act=${act.toFixed(4)} | inW=${n.in_weight_l2?.toFixed(3) ?? '—'}`}
          >
            <span style={{ width: 32, flexShrink: 0, color: colors.inkMuted48, textAlign: 'right', fontSize: 9 }}>N{n.neuron_index}</span>
            <div style={{ flex: 1, background: colors.canvas, borderRadius: 2, height: 10, overflow: 'hidden' }}>
              <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2, transition: 'width 0.15s' }} />
            </div>
            <span style={{ width: 42, flexShrink: 0, fontFamily: 'monospace', fontSize: 8, color: colors.bodyMuted }}>
              {act.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ---------- Main Component ---------- */

interface Props {
  onNavigate?: (page: string) => void;
}

export default function Gpt2NeuronExplorer({ onNavigate }: Props) {
  // Model state
  const [modelInfo, setModelInfo] = useState<{ num_layers: number; num_heads: number; hidden_dim: number; vocab_size: number } | null>(null);
  const [architecture, setArchitecture] = useState<any>(null);
  const [loadingModel, setLoadingModel] = useState(false);

  // Prompt state
  const [prompt, setPrompt] = useState('The capital of France is');
  const [promptResult, setPromptResult] = useState<any>(null);
  const [runningPrompt, setRunningPrompt] = useState(false);

  // Layer/head/neuron selection
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedNeuron, setSelectedNeuron] = useState<number | null>(null);
  const [selectedComponent, setSelectedComponent] = useState<'mlp' | 'resid'>('mlp');

  // Fetched data
  const [attentionPattern, setAttentionPattern] = useState<any>(null);
  const [layerDetail, setLayerDetail] = useState<any>(null);
  const [neuronList, setNeuronList] = useState<any[]>([]);
  const [neuronDetail, setNeuronDetail] = useState<any>(null);
  const [attentionLoading, setAttentionLoading] = useState(false);
  const [neuronLoading, setNeuronLoading] = useState(false);
  const [layerLoading, setLayerLoading] = useState(false);

  const numLayers = modelInfo?.num_layers ?? 12;
  const numHeads = modelInfo?.num_heads ?? 12;
  const dModel = modelInfo?.hidden_dim ?? 768;
  const dMlp = architecture?.layers?.[0]?.num_mlp_neurons ?? 3072;

  // Load model
  const handleLoadModel = useCallback(async () => {
    setLoadingModel(true);
    try {
      const result = await api.gpt2Load('gpt2');
      setModelInfo({
        num_layers: result.n_layers ?? 12,
        num_heads: result.n_heads ?? 12,
        hidden_dim: result.d_model ?? 768,
        vocab_size: result.vocab_size ?? 50257,
      });
      // Fetch full architecture dynamically
      try {
        const arch = await api.pythonCall('gpt2/architecture', {});
        if (arch.status === 'ok') {
          setArchitecture(arch);
        }
      } catch {}
    } catch (e: any) {
      setModelInfo({ num_layers: 12, num_heads: 12, hidden_dim: 768, vocab_size: 50257 });
    } finally {
      setLoadingModel(false);
    }
  }, []);

  // Run prompt
  const handleRunPrompt = useCallback(async () => {
    if (!prompt.trim()) return;
    setRunningPrompt(true);
    setPromptResult(null);
    setAttentionPattern(null);
    setNeuronDetail(null);
    setSelectedNeuron(null);
    try {
      const result = await api.gpt2RunPrompt(prompt);
      setPromptResult(result);
      // Auto-fetch attention for current layer/head
      fetchAttention(selectedLayer, selectedHead);
      // Auto-fetch layer detail
      fetchLayerDetail(selectedLayer);
      // Auto-fetch neuron list
      fetchNeuronList(selectedLayer, selectedComponent);
    } catch (e: any) {
      setPromptResult({ status: 'error', error: e.message });
    } finally {
      setRunningPrompt(false);
    }
  }, [prompt, selectedLayer, selectedHead, selectedComponent]);

  // Fetch attention pattern
  const fetchAttention = useCallback(async (layer: number, head: number) => {
    setAttentionLoading(true);
    try {
      const result = await api.gpt2AttentionHead(layer, head);
      setAttentionPattern(result);
    } catch {
      setAttentionPattern(null);
    } finally {
      setAttentionLoading(false);
    }
  }, []);

  // Fetch layer detail
  const fetchLayerDetail = useCallback(async (layer: number) => {
    setLayerLoading(true);
    try {
      const result = await api.pythonCall('gpt2/layer', { layer });
      if (result.status === 'ok') {
        setLayerDetail(result);
      }
    } catch {
      setLayerDetail(null);
    } finally {
      setLayerLoading(false);
    }
  }, []);

  // Fetch neuron list
  const fetchNeuronList = useCallback(async (layer: number, component: string) => {
    setNeuronLoading(true);
    try {
      const result = await api.pythonCall('gpt2/neurons', {
        layer, component, page: 0, page_size: 128, sort_by: 'activation', order: 'desc',
      });
      if (result.status === 'ok' && result.neurons) {
        setNeuronList(result.neurons);
      }
    } catch {
      setNeuronList([]);
    } finally {
      setNeuronLoading(false);
    }
  }, []);

  // Fetch neuron detail (when clicked)
  const fetchNeuronDetail = useCallback(async (layer: number, neuronIndex: number, component: string) => {
    setNeuronLoading(true);
    try {
      const result = await api.pythonCall('gpt2/neuron', {
        layer, neuron_index: neuronIndex, component, top_k_weights: 16,
      });
      if (result.status === 'ok') {
        setNeuronDetail(result);
      }
    } catch {
      setNeuronDetail(null);
    } finally {
      setNeuronLoading(false);
    }
  }, []);

  // Handlers for selection changes
  const handleLayerChange = useCallback((layer: number) => {
    setSelectedLayer(layer);
    setSelectedNeuron(null);
    setNeuronDetail(null);
    fetchAttention(layer, selectedHead);
    fetchLayerDetail(layer);
    fetchNeuronList(layer, selectedComponent);
  }, [selectedHead, selectedComponent, fetchAttention, fetchLayerDetail, fetchNeuronList]);

  const handleHeadChange = useCallback((head: number) => {
    setSelectedHead(head);
    setAttentionPattern(null);
    fetchAttention(selectedLayer, head);
  }, [selectedLayer, fetchAttention]);

  const handleNeuronSelect = useCallback((neuronIndex: number) => {
    setSelectedNeuron(neuronIndex);
    setNeuronDetail(null);
    fetchNeuronDetail(selectedLayer, neuronIndex, selectedComponent);
  }, [selectedLayer, selectedComponent, fetchNeuronDetail]);

  const handleComponentChange = useCallback((comp: 'mlp' | 'resid') => {
    setSelectedComponent(comp);
    setSelectedNeuron(null);
    setNeuronDetail(null);
    setNeuronList([]);
    fetchNeuronList(selectedLayer, comp);
  }, [selectedLayer, fetchNeuronList]);

  // Auto-load model on mount
  useEffect(() => {
    handleLoadModel();
  }, [handleLoadModel]);

  const loaded = modelInfo !== null;
  const hasPrompt = promptResult?.status === 'ok';
  const top16 = promptResult?.top16 || [];
  const maxLogit = top16.length ? Math.max(...top16.map((t: any) => Math.abs(t.logit)), 0.01) : 0.01;

  const umapPoints = useMemo(
    () => buildLayerNeuronPoints(neuronList, selectedLayer),
    [neuronList, selectedLayer],
  );

  return (
    <div style={{ padding: '16px 20px', maxWidth: 1200, display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>GPT-2 Neuron Explorer</h2>
          <p style={{ margin: '4px 0 0', fontSize: 12, color: colors.inkMuted48 }}>
            Dynamic visualization — every neuron and layer, driven by the live GPT-2 model via the Electron backend
          </p>
        </div>
        <button className="btn btn-primary" onClick={handleLoadModel} disabled={loadingModel}>
          {loadingModel ? <>Loading<Spinner /></> : loaded ? 'Reload Model' : 'Load GPT-2'}
        </button>
      </div>

      {/* Model Info Bar — dynamic, from model config */}
      {modelInfo && (
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, background: colors.surfacePearl, padding: 10, borderRadius: 6 }}>
          <KV k="Layers" v={modelInfo.num_layers} />
          <KV k="Heads" v={modelInfo.num_heads} />
          <KV k="d_model" v={modelInfo.hidden_dim} />
          <KV k="d_mlp" v={architecture?.layers?.[0]?.num_mlp_neurons ?? dMlp} />
          <KV k="Vocab" v={modelInfo.vocab_size} />
          <KV k="Model" v="gpt2" mono />
          <KV k="Status" v={loaded ? 'loaded' : 'loading'}><StatusBadge value={loaded ? 'loaded' : 'loading'} /></KV>
        </div>
      )}

      {/* Prompt Input */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <input
          className="input-text"
          style={{ flex: 1 }}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleRunPrompt()}
          placeholder="Enter a prompt..."
        />
        <button className="btn btn-primary" onClick={handleRunPrompt} disabled={runningPrompt || !loaded}>
          {runningPrompt ? <>Running<Spinner /></> : 'Run Prompt'}
        </button>
        {promptResult?.status === 'ok' && (
          <span style={{ fontSize: 12, color: colors.success }}>
            Top-1: "{promptResult.top5?.[0]?.token}" | Next: "{promptResult.next_token}"
          </span>
        )}
      </div>

      {/* Top predictions */}
      {hasPrompt && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {top16.map((t: any, i: number) => (
            <span key={i} style={{
              padding: '2px 6px', borderRadius: 3, fontSize: 11,
              background: i === 0 ? rgba(colors.primary, 0.2) : colors.surfacePearl,
              fontFamily: 'monospace', fontWeight: i === 0 ? 700 : 400,
              border: i === 0 ? `1px solid ${colors.primary}` : '1px solid transparent',
            }}>
              "{t.token}" {t.logit.toFixed(2)}
            </span>
          ))}
        </div>
      )}

      {/* Main visualization area — 3-column layout */}
      {loaded && hasPrompt && (
        <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr 300px', gap: 10, minHeight: 500 }}>
          {/* LEFT: Layer + Neuron list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, overflow: 'auto', maxHeight: 600 }}>
            {/* Layer selector — dynamic count from model */}
            <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                Layers ({numLayers})
              </div>
              {Array.from({ length: numLayers }, (_, i) => (
                <div
                  key={i}
                  onClick={() => handleLayerChange(i)}
                  style={{
                    padding: '3px 6px', cursor: 'pointer', borderRadius: 3, fontSize: 10,
                    background: selectedLayer === i ? rgba(colors.primary, 0.15) : 'transparent',
                    border: selectedLayer === i ? `1px solid ${colors.primary}` : '1px solid transparent',
                    marginBottom: 1, display: 'flex', justifyContent: 'space-between',
                  }}
                >
                  <span>L{i} (blocks.{i})</span>
                  <span style={{ color: colors.inkMuted48, fontFamily: 'monospace', fontSize: 9 }}>
                    {architecture?.layers?.[i]?.num_mlp_neurons ?? dMlp}N
                  </span>
                </div>
              ))}
            </div>

            {/* Component selector */}
            <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                Component
              </div>
              <div style={{ display: 'flex', gap: 4 }}>
                <button
                  className={clsx('btn', selectedComponent === 'mlp' ? 'btn-primary' : 'btn-secondary')}
                  onClick={() => handleComponentChange('mlp')}
                  style={{ fontSize: 10, padding: '3px 6px' }}
                >
                  MLP ({dMlp})
                </button>
                <button
                  className={clsx('btn', selectedComponent === 'resid' ? 'btn-primary' : 'btn-secondary')}
                  onClick={() => handleComponentChange('resid')}
                  style={{ fontSize: 10, padding: '3px 6px' }}
                >
                  Resid ({dModel})
                </button>
              </div>
            </div>

            {/* Neuron list — dynamic count from model */}
            <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8, flex: 1, overflow: 'auto' }}>
              <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                Neurons (click to inspect)
              </div>
              {neuronLoading ? (
                <div style={{ fontSize: 10, color: colors.inkMuted48 }}>Loading...</div>
              ) : (
                <NeuronBarChart
                  neurons={neuronList}
                  selectedNeuron={selectedNeuron}
                  onSelect={handleNeuronSelect}
                />
              )}
            </div>
          </div>

          {/* CENTER: Attention heatmap + layer detail */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, overflow: 'auto', maxHeight: 600 }}>
            {/* Attention heatmap — dynamic head count from model */}
            <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1 }}>
                  Attention L{selectedLayer}H{selectedHead}
                </span>
                <select
                  value={selectedHead}
                  onChange={(e) => handleHeadChange(Number(e.target.value))}
                  style={{ fontSize: 10, background: colors.canvas, color: colors.inkMuted48, border: `1px solid ${colors.hairline}`, borderRadius: 3, padding: '2px 4px' }}
                >
                  {Array.from({ length: numHeads }, (_, i) => (
                    <option key={i} value={i}>H{i}</option>
                  ))}
                </select>
                {attentionLoading && <Spinner />}
              </div>
              {attentionPattern?.status === 'ok' ? (
                <AttentionHeatmap matrix={attentionPattern.matrix} tokens={attentionPattern.str_tokens || []} />
              ) : (
                <p className="hint" style={{ fontSize: 10 }}>Run a prompt to see attention patterns</p>
              )}
            </div>

            {/* Layer detail — from live model weights */}
            {layerLoading && <div style={{ fontSize: 10, color: colors.inkMuted48, padding: 4 }}>Loading layer detail...</div>}
            {layerDetail?.status === 'ok' && (
              <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                  Layer {layerDetail.layer} Detail (from live weights)
                </div>
                <KV k="MLP dim" v={layerDetail.mlp?.d_mlp} />
                <KV k="c_fc shape" v={layerDetail.mlp?.c_fc_shape?.join('×')} mono />
                <KV k="c_proj shape" v={layerDetail.mlp?.c_proj_shape?.join('×')} mono />
                <KV k="In-weight L2 mean" v={layerDetail.mlp?.in_weight_l2_stats?.mean?.toFixed(4) ?? '—'} mono />
                <KV k="Out-weight L2 mean" v={layerDetail.mlp?.out_weight_l2_stats?.mean?.toFixed(4) ?? '—'} mono />
                <KV k="Params" v={layerDetail.n_params?.toLocaleString()} mono />
                {layerDetail.top_active_neurons?.length > 0 && (
                  <div style={{ marginTop: 6 }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, marginBottom: 3 }}>Top Active Neurons</div>
                    {layerDetail.top_active_neurons.slice(0, 6).map((n: any) => (
                      <div key={n.neuron_index} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, padding: '1px 0' }}>
                        <span>N{n.neuron_index}</span>
                        <span style={{ fontFamily: 'monospace', color: colors.success }}>{n.activation.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Attention heads summary */}
            {layerDetail?.attention_heads && (
              <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                  Attention Heads (from live weights)
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                  {layerDetail.attention_heads.map((h: any) => (
                    <div
                      key={h.head_index}
                      onClick={() => handleHeadChange(h.head_index)}
                      style={{
                        padding: '2px 5px', borderRadius: 3, fontSize: 9, cursor: 'pointer',
                        background: selectedHead === h.head_index ? rgba(colors.primary, 0.15) : colors.surfacePearl,
                        border: selectedHead === h.head_index ? `1px solid ${colors.primary}` : '1px solid transparent',
                      }}
                      title={`H${h.head_index}: Q=${h.q_weight_l2?.toFixed(3)} K=${h.k_weight_l2?.toFixed(3)}`}
                    >
                      H{h.head_index}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* RIGHT: Neuron inspector — detailed info when clicked */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, overflow: 'auto', maxHeight: 600 }}>
            {neuronLoading && !neuronDetail && (
              <div style={{ fontSize: 10, color: colors.inkMuted48, padding: 6 }}>Inspecting neuron...</div>
            )}
            {neuronDetail?.status === 'ok' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {/* Header */}
                <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
                  <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                    Neuron {neuronDetail.id}
                  </div>
                  <KV k="Path" v={neuronDetail.path} mono />
                  <KV k="Component" v={neuronDetail.component} />
                  <KV k="d_model" v={neuronDetail.d_model} mono />
                  <KV k="d_mlp" v={neuronDetail.d_mlp} mono />
                  <KV k="Bias" v={neuronDetail.bias?.toFixed(4) ?? '—'} mono />
                </div>

                {/* Weight statistics — from live model weights */}
                <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
                  <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                    Weight Statistics (live)
                  </div>
                  <KV k="In-weight L2" v={neuronDetail.in_weight_l2?.toFixed(4) ?? '—'} mono />
                  <KV k="Out-weight L2" v={neuronDetail.out_weight_l2?.toFixed(4) ?? '—'} mono />
                  {neuronDetail.in_weight_stats && (
                    <>
                      <KV k="In mean" v={neuronDetail.in_weight_stats.mean?.toFixed(6) ?? '—'} mono />
                      <KV k="In std" v={neuronDetail.in_weight_stats.std?.toFixed(6) ?? '—'} mono />
                      <KV k="In range" v={`[${neuronDetail.in_weight_stats.min?.toFixed(4)}, ${neuronDetail.in_weight_stats.max?.toFixed(4)}]`} mono />
                    </>
                  )}
                  {neuronDetail.out_weight_stats && (
                    <>
                      <KV k="Out mean" v={neuronDetail.out_weight_stats.mean?.toFixed(6) ?? '—'} mono />
                      <KV k="Out std" v={neuronDetail.out_weight_stats.std?.toFixed(6) ?? '—'} mono />
                    </>
                  )}
                </div>

                {/* Top input weights */}
                {neuronDetail.top_input_weights_positive?.length > 0 && (
                  <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                      Top Input Weights (positive)
                    </div>
                    {neuronDetail.top_input_weights_positive.slice(0, 6).map((w: any, i: number) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, marginBottom: 1 }}>
                        <span style={{ color: colors.inkMuted48 }}>dim {w.dim}</span>
                        <span style={{ fontFamily: 'monospace', color: colors.primary }}>{w.weight.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Top output weights */}
                {neuronDetail.top_output_weights_positive?.length > 0 && (
                  <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                      Top Output Weights (positive)
                    </div>
                    {neuronDetail.top_output_weights_positive.slice(0, 6).map((w: any, i: number) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, marginBottom: 1 }}>
                        <span style={{ color: colors.inkMuted48 }}>dim {w.dim}</span>
                        <span style={{ fontFamily: 'monospace', color: colors.success }}>{w.weight.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Per-token activations */}
                {neuronDetail.per_token_activations?.length > 0 && (
                  <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                      Per-Token Activations
                    </div>
                    <div style={{ maxHeight: 100, overflowY: 'auto' }}>
                      {neuronDetail.per_token_activations.map((ta: any, i: number) => (
                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, marginBottom: 1 }}>
                          <span style={{ fontFamily: 'monospace', color: colors.bodyMuted }}>{ta.token}</span>
                          <span style={{ fontFamily: 'monospace', color: ta.activation >= 0 ? colors.primary : colors.danger }}>
                            {ta.activation.toFixed(4)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Activation stats */}
                {neuronDetail.activation_stats && (
                  <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                      Activation Stats
                    </div>
                    <KV k="Mean" v={neuronDetail.activation_stats.mean?.toFixed(6) ?? '—'} mono />
                    <KV k="Std" v={neuronDetail.activation_stats.std?.toFixed(6) ?? '—'} mono />
                    <KV k="Min" v={neuronDetail.activation_stats.min?.toFixed(6) ?? '—'} mono />
                    <KV k="Max" v={neuronDetail.activation_stats.max?.toFixed(6) ?? '—'} mono />
                    <KV k="Sparsity" v={neuronDetail.activation_stats.sparsity?.toFixed(4) ?? '—'} mono />
                  </div>
                )}

                {/* Nearest neurons */}
                {neuronDetail.nearest_neurons?.length > 0 && (
                  <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
                    <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
                      Nearest Neurons (cosine sim)
                    </div>
                    {neuronDetail.nearest_neurons.map((n: any, i: number) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, marginBottom: 1 }}>
                        <span style={{ color: colors.inkMuted48 }}>L{n.layer}.N{n.neuron_index}</span>
                        <span style={{ fontFamily: 'monospace', color: colors.purple }}>{n.similarity.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Description */}
                <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8, fontSize: 10, color: colors.bodyMuted, lineHeight: 1.5 }}>
                  {neuronDetail.description}
                </div>
              </div>
            )}

            {!selectedNeuron && (
              <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8, fontSize: 10, color: colors.inkMuted48 }}>
                Click a neuron in the list to inspect its weights, activations, and connections from the live model.
              </div>
            )}
          </div>

        {/* Neuron UMAP — current layer */}
        {neuronList.length > 0 && (
          <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 8 }}>
            <div style={{ fontSize: 10, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>
              Neuron Map — Layer {selectedLayer} ({selectedComponent})
            </div>
            <NeuronUMAP
              points={umapPoints}
              selectedId={selectedNeuron !== null ? idForLayerNeuron(selectedLayer, selectedNeuron) : null}
              onSelectNeuron={id => {
                const parsed = parseNeuronId(id);
                if (parsed && parsed.layer === selectedLayer) {
                  handleNeuronSelect(parsed.neuron);
                } else {
                  setSelectedNeuron(null);
                  setNeuronDetail(null);
                }
              }}
              height={420}
            />
          </div>
        )}
        </div>
      )}

      {/* Architecture tree — dynamic, from live model */}
      {loaded && !hasPrompt && architecture && (
        <div style={{ background: colors.surfacePearl, borderRadius: 6, padding: 10, overflow: 'auto', maxHeight: 400 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: colors.inkMuted48, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
            Model Architecture (Dynamic — from live model weights)
          </div>
          {architecture.modules?.map((mod: any) => (
            <div key={mod.id} style={{ marginBottom: 6, fontSize: 11 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                <span style={{ fontFamily: 'monospace', color: colors.primary }}>{mod.path || mod.id}</span>
                <span style={{ color: colors.inkMuted48 }}>{mod.type}</span>
                {mod.shape && <span style={{ color: colors.inkMuted48, fontFamily: 'monospace' }}>{mod.shape.join('×')}</span>}
                {mod.n_params && <span style={{ color: colors.inkMuted48 }}>{mod.n_params.toLocaleString()} params</span>}
              </div>
              {mod.layers && (
                <div style={{ marginLeft: 16, marginTop: 4 }}>
                  {mod.layers.map((l: any) => (
                    <div key={l.index} style={{ display: 'flex', gap: 8, fontSize: 10, padding: '2px 0', color: colors.bodyMuted }}>
                      <span style={{ fontFamily: 'monospace' }}>blocks.{l.index}</span>
                      <span>{l.num_attention_heads} heads</span>
                      <span>{l.num_mlp_neurons} MLP neurons</span>
                      <span>{l.d_model} dim</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {!loaded && (
        <div style={{ textAlign: 'center', padding: 40, color: colors.inkMuted48 }}>
          Load GPT-2 to start exploring neurons and layers dynamically.
        </div>
      )}
    </div>
  );
}

function clsx(...classes: (string | false | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}
