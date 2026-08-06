import React, { useState, useCallback, useEffect, useRef } from 'react';
import { api } from '../services/api';
import { colors } from '../design/tokens';

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rgba(hex: string, a: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface GraphNode {
  id: string;
  label: string;
  x: number;
  y: number;
  type: 'embedding' | 'layernorm' | 'attention-head' | 'mlp-neuron' | 'mlp-group' | 'output' | 'block-input' | 'block-output';
  layer: number;
  blockIndex?: number;
  headIndex?: number;
  neuronIndex?: number;
  nParams?: number;
  dim?: number;
  activation?: number;
  absActivation?: number;
  inWeightL2?: number;
  outWeightL2?: number;
}

interface GraphEdge {
  from: string;
  to: string;
  type: 'feedforward' | 'residual' | 'attention';
  weight?: number;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface Props {
  onNavigate?: (page: string) => void;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const COLORS = {
  bg: colors.canvasParchment,
  nodeEmbedding: colors.primary,
  nodeLayernorm: colors.purple,
  nodeAttention: colors.warning,
  nodeMlp: colors.success,
  nodeOutput: colors.pink,
  nodeBlockInput: colors.primary,
  nodeBlockOutput: colors.primary,
  edgeFeedforward: colors.hairline,
  edgeResidual: colors.warning,
  edgeAttention: colors.purple,
  edgeHighlight: colors.purpleBorder,
  text: colors.ink,
  textDim: colors.inkMuted48,
  accent: colors.purpleBorder,
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
};

const VIRIDIS = COLORS.viridis;
const BLOCK_W = 180;
const NODE_R = 6;
const LAYER_GAP = 160;
const NODE_GAP = 18;

  /* ------------------------------------------------------------------ */
  /*  Helpers                                                            */
  /* ------------------------------------------------------------------ */

const fmt = (n: number | undefined | null): string => {
  if (n === undefined || n === null || Number.isNaN(n)) return '—';
  if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(0) + 'K';
  return String(Math.round(n * 100) / 100);
};

function ramp(t: number, stops: number[][]): string {
  const tt = Math.max(0, Math.min(1, t));
  const scaled = tt * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const [r, g, b] = stops[i].map((ch, k) => Math.round(ch + (stops[i + 1][k] - ch) * f));
  return `rgb(${r},${g},${b})`;
}

function nodeColor(node: GraphNode): string {
  if (node.type === 'embedding' || node.type === 'block-input') return COLORS.nodeEmbedding;
  if (node.type === 'layernorm') return COLORS.nodeLayernorm;
  if (node.type === 'attention-head') return COLORS.nodeAttention;
  if (node.type === 'mlp-neuron' || node.type === 'mlp-group') return COLORS.nodeMlp;
  if (node.type === 'output') return COLORS.nodeOutput;
  return COLORS.textDim;
}

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

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
      <span style={{ color: colors.inkMuted48, minWidth: 140, flexShrink: 0 }}>{k}</span>
      <span style={mono ? { fontFamily: 'monospace' } : {}}>{v ?? '—'}</span>
    </div>
  );
}

function MiniBar({ label, value, max, color = colors.purpleBorder }: { label: string; value: number; max: number; color?: string }) {
  const pct = max > 0 ? Math.max(2, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, marginBottom: 2 }}>
      <span style={{ width: 34, color: colors.inkMuted48, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, background: colors.canvas, borderRadius: 2, height: 6, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2 }} />
      </div>
      <span style={{ width: 44, textAlign: 'right', fontFamily: 'monospace', color: colors.bodyMuted, flexShrink: 0 }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */

export default function TransformerVisualizer({ onNavigate }: Props) {
  const [arch, setArch] = useState<any>(null);
  const [archLoading, setArchLoading] = useState(true);
  const [loadingModel, setLoadingModel] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState('The capital of France is');
  const [promptRunning, setPromptRunning] = useState(false);
  const [promptStatus, setPromptStatus] = useState<string | null>(null);

  // Graph state
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [nodeDetails, setNodeDetails] = useState<any>(null);
  const [nodeLoading, setNodeLoading] = useState(false);
  const [activations, setActivations] = useState<Map<string, number>>(new Map());

  // View state
  const [zoom, setZoom] = useState(0.6);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const svgRef = useRef<SVGSVGElement>(null);

  /* ---------------- data loading ---------------- */

  const loadArchitecture = useCallback(async () => {
    setArchLoading(true);
    setError(null);
    try {
      const a = await api.gpt2Architecture();
      setArch(a);
    } catch (e: any) {
      setError(`Cannot reach backend (${e.message}). Start the Python server on :8000 first.`);
    } finally {
      setArchLoading(false);
    }
  }, []);

  useEffect(() => { void loadArchitecture(); }, [loadArchitecture]);

  const loadModel = useCallback(async () => {
    setLoadingModel(true);
    setError(null);
    try {
      const r = await api.gpt2Load('gpt2');
      if (r?.status !== 'loaded' && r?.status !== 'ok') throw new Error(r?.error || 'Model load failed');
      await loadArchitecture();
    } catch (e: any) {
      setError(`Model load failed: ${e.message}`);
    } finally {
      setLoadingModel(false);
    }
  }, [loadArchitecture]);

  const buildGraph = useCallback((archData: any): GraphData => {
    const nodes: GraphNode[] = [];
    const edges: GraphEdge[] = [];
    const layers = archData.layers || [];
    const modules = archData.modules || [];

    // Column 0: WTE + WPE (input)
    const wte = modules.find((m: any) => m.id === 'wte');
    const wpe = modules.find((m: any) => m.id === 'wpe');
    const wteNode: GraphNode = {
      id: 'wte', label: 'WTE', x: 0, y: 0,
      type: 'embedding', layer: 0,
      nParams: wte?.n_params, dim: wte?.shape?.[1],
    };
    const wpeNode: GraphNode = {
      id: 'wpe', label: 'WPE', x: 0, y: 60,
      type: 'embedding', layer: 0,
      nParams: wpe?.n_params, dim: wpe?.shape?.[1],
    };
    nodes.push(wteNode, wpeNode);

    // Columns 1-12: 12 transformer blocks
    layers.forEach((layer: any, li: number) => {
      const colX = 1 + li;
      const blockBaseY = 0;

      // Block input node
      const blockInput: GraphNode = {
        id: `block-${li}-in`, label: `B${li} in`, x: colX, y: blockBaseY,
        type: 'block-input', layer: li,
        nParams: layer.n_params,
      };
      nodes.push(blockInput);

      // Edge: WTE/WPE → block input (residual)
      edges.push({ from: 'wte', to: `block-${li}-in`, type: 'residual' });
      edges.push({ from: 'wpe', to: `block-${li}-in`, type: 'residual' });

      // LN1
      const ln1 = layer.components.find((c: any) => c.id === 'ln_1');
      const ln1Node: GraphNode = {
        id: `block-${li}-ln1`, label: `LN₁`, x: colX, y: blockBaseY + 50,
        type: 'layernorm', layer: li, dim: ln1?.dim,
      };
      nodes.push(ln1Node);
      edges.push({ from: `block-${li}-in`, to: `block-${li}-ln1`, type: 'feedforward' });

      // Attention heads
      const attn = layer.components.find((c: any) => c.type === 'attention');
      const nHeads = attn?.n_heads || 12;
      const headNodes: GraphNode[] = [];
      for (let h = 0; h < nHeads; h++) {
        const headNode: GraphNode = {
          id: `block-${li}-head-${h}`, label: `H${h}`, x: colX, y: blockBaseY + 90 + h * NODE_GAP,
          type: 'attention-head', layer: li, headIndex: h,
          dim: attn?.d_head,
        };
        nodes.push(headNode);
        headNodes.push(headNode);
        edges.push({ from: `block-${li}-ln1`, to: `block-${li}-head-${h}`, type: 'attention' });
      }

      // LN2
      const ln2 = layer.components.find((c: any) => c.id === 'ln_2');
      const ln2Node: GraphNode = {
        id: `block-${li}-ln2`, label: `LN₂`, x: colX, y: blockBaseY + 90 + nHeads * NODE_GAP + 30,
        type: 'layernorm', layer: li, dim: ln2?.dim,
      };
      nodes.push(ln2Node);
      // Residual from block input to LN2
      edges.push({ from: `block-${li}-in`, to: `block-${li}-ln2`, type: 'residual' });
      // Attention output → LN2
      edges.push({ from: `block-${li}-head-${nHeads - 1}`, to: `block-${li}-ln2`, type: 'feedforward' });

      // MLP neurons (sampled)
      const mlp = layer.components.find((c: any) => c.type === 'mlp');
      const nMlpNeurons = mlp?.num_neurons || 3072;
      const mlpSampleSize = Math.min(64, nMlpNeurons);
      const mlpStartY = blockBaseY + 90 + nHeads * NODE_GAP + 60;
      const mlpNodes: GraphNode[] = [];
      for (let i = 0; i < mlpSampleSize; i++) {
        const neuronNode: GraphNode = {
          id: `block-${li}-mlp-${i}`, label: `N${i}`, x: colX, y: mlpStartY + i * (NODE_GAP * 0.6),
          type: 'mlp-neuron', layer: li, neuronIndex: i,
          nParams: 1, // placeholder
        };
        nodes.push(neuronNode);
        mlpNodes.push(neuronNode);
        edges.push({ from: `block-${li}-ln2`, to: `block-${li}-mlp-${i}`, type: 'feedforward' });
      }

      // Block output node
      const blockOutput: GraphNode = {
        id: `block-${li}-out`, label: `B${li} out`, x: colX, y: mlpStartY + mlpSampleSize * (NODE_GAP * 0.6) + 30,
        type: 'block-output', layer: li,
        nParams: layer.n_params,
      };
      nodes.push(blockOutput);
      // MLP → block output
      if (mlpNodes.length > 0) {
        edges.push({ from: `block-${li}-mlp-${mlpNodes.length - 1}`, to: `block-${li}-out`, type: 'feedforward' });
      }
      // Residual: block input → block output
      edges.push({ from: `block-${li}-in`, to: `block-${li}-out`, type: 'residual' });

      // Edge to next block
      if (li < layers.length - 1) {
        const nextColX = 1 + li + 1;
        edges.push({ from: `block-${li}-out`, to: `block-${li + 1}-in`, type: 'feedforward' });
      }
    });

    // Final LayerNorm
    const lnF = modules.find((m: any) => m.id === 'ln_f');
    const lnFNode: GraphNode = {
      id: 'ln-f', label: 'LN_f', x: 1 + layers.length, y: 0,
      type: 'layernorm', layer: layers.length, dim: lnF?.dim,
      nParams: lnF?.n_params,
    };
    nodes.push(lnFNode);
    // Last block output → LN_f
    if (layers.length > 0) {
      edges.push({ from: `block-${layers.length - 1}-out`, to: 'ln-f', type: 'feedforward' });
    }

    // LM Head
    const lmHead = modules.find((m: any) => m.id === 'lm_head');
    const lmHeadNode: GraphNode = {
      id: 'lm-head', label: 'LM Head', x: 2 + layers.length, y: 0,
      type: 'output', layer: layers.length + 1,
      nParams: lmHead?.n_params, dim: lmHead?.shape?.[1],
    };
    nodes.push(lmHeadNode);
    edges.push({ from: 'ln-f', to: 'lm-head', type: 'feedforward' });

    return { nodes, edges };
  }, []);

  useEffect(() => {
    if (arch?.status === 'ok') {
      const gd = buildGraph(arch);
      setGraphData(gd);
      // Center the graph
      const maxX = Math.max(...gd.nodes.map((n) => n.x), 1);
      const maxY = Math.max(...gd.nodes.map((n) => n.y), 1);
      setPan({ x: 400, y: 300 });
    }
  }, [arch, buildGraph]);

  const loadNodeDetail = useCallback(async (nodeId: string) => {
    setSelectedNode(nodeId);
    setNodeLoading(true);
    setNodeDetails(null);
    try {
      const parts = nodeId.split('-');
      // Parse node id: block-{li}-head-{h}, block-{li}-mlp-{i}, etc.
      if (nodeId.startsWith('block-')) {
        const li = parseInt(parts[1]);
        if (nodeId.includes('head')) {
          const h = parseInt(parts[3]);
          const d = await api.gpt2AttentionHead(li, h);
          setNodeDetails(d);
        } else if (nodeId.includes('mlp')) {
          const ni = parseInt(parts[3]);
          const d = await api.gpt2Neuron(li, ni, 'mlp', 10);
          setNodeDetails(d);
        } else {
          const d = await api.gpt2Layer(li);
          setNodeDetails(d);
        }
      } else if (nodeId === 'ln-f') {
        const d = await api.gpt2Layer(0); // just to get module info
        setNodeDetails(d);
      } else {
        setNodeDetails(null);
      }
    } catch (e: any) {
      setNodeDetails(null);
    } finally {
      setNodeLoading(false);
    }
  }, []);

  const runPrompt = useCallback(async () => {
    if (!prompt.trim() || promptRunning) return;
    setPromptRunning(true);
    setPromptStatus(null);
    setError(null);
    try {
      const r = await api.gpt2RunPrompt(prompt.trim());
      if (r?.status !== 'ok' && r?.status !== 'loaded') throw new Error(r?.error || 'Prompt run failed');
      setPromptStatus(`ran ${prompt.trim().split(/\s+/).length} words · tokens cached`);
      // Refresh graph with activation data
      if (arch?.status === 'ok') {
        const gd = buildGraph(arch);
        setGraphData(gd);
      }
    } catch (e: any) {
      setError(`Prompt run failed: ${e.message}`);
    } finally {
      setPromptRunning(false);
    }
  }, [prompt, promptRunning, arch, buildGraph]);

  /* ---------------- render ---------------- */

  const loaded = arch?.status === 'ok';
  const layers = arch?.layers || [];
  const modules = arch?.modules || [];
  const nLayers = layers.length;

  function BlockCard({ layer, idx }: { layer: any; idx: number }) {
    const comps = layer.components || [];
    const ln1 = comps.find((c: any) => c.id === 'ln_1');
    const attn = comps.find((c: any) => c.type === 'attention');
    const ln2 = comps.find((c: any) => c.id === 'ln_2');
    const mlp = comps.find((c: any) => c.type === 'mlp');
    const nHeads = attn?.n_heads || 12;

    return (
      <div style={{
        display: 'flex', flexDirection: 'column', gap: 6,
        width: 160, flexShrink: 0,
        padding: 10, borderRadius: 8,
        background: colors.surface, border: `1px solid ${colors.hairline}`,
      }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: colors.inkMuted48, textAlign: 'center' }}>
          Block {idx}
        </div>

        <div style={{
          padding: 6, borderRadius: 4,
          background: colors.canvas, border: `1px solid ${colors.hairline}`,
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 9, color: colors.inkMuted48 }}>LN₁</div>
          <div style={{ fontSize: 8, color: colors.bodyMuted }}>{ln1?.dim ?? '—'}d</div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2, justifyContent: 'center' }}>
          {Array.from({ length: nHeads }).map((_, h) => {
            const headId = `block-${idx}-head-${h}`;
            const isSelected = selectedNode === headId;
            return (
              <div
                key={h}
                onClick={() => { setSelectedNode(headId); void loadNodeDetail(headId); }}
                style={{
                  width: 14, height: 14, borderRadius: 2,
                  background: isSelected ? colors.purpleBorder : colors.warning,
                  color: isSelected ? colors.onDark : colors.ink,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 7, fontWeight: 600, cursor: 'pointer',
                  border: `1px solid ${isSelected ? colors.onDark : colors.hairline}`,
                }}
                title={`Head ${h}`}
              >
                {h}
              </div>
            );
          })}
        </div>

        <div style={{
          padding: 6, borderRadius: 4,
          background: colors.canvas, border: `1px solid ${colors.hairline}`,
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 9, color: colors.inkMuted48 }}>LN₂</div>
          <div style={{ fontSize: 8, color: colors.bodyMuted }}>{ln2?.dim ?? '—'}d</div>
        </div>

        <div style={{ display: 'flex', gap: 1, height: 60, alignItems: 'flex-end' }}>
          {Array.from({ length: 8 }).map((_, i) => {
            const nid = `block-${idx}-mlp-${i}`;
            return (
              <div
                key={i}
                onClick={() => { setSelectedNode(nid); void loadNodeDetail(nid); }}
                style={{
                  flex: 1, minHeight: 2,
                  background: selectedNode === nid ? colors.primary : colors.inkMuted48,
                  opacity: 0.4, borderRadius: 1, cursor: 'pointer',
                }}
                title={`N${i}`}
              />
            );
          })}
        </div>

        <div style={{ fontSize: 8, color: colors.bodyMuted, textAlign: 'center' }}>
          {fmt(layer.n_params)} params
        </div>
      </div>
    );
  }

  return (
    <div data-testid="transformer-visualizer" style={{ padding: 16, height: '100%', display: 'flex', flexDirection: 'column', background: COLORS.bg }}>
      {/* Header + prompt bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
        <h2 style={{ margin: 0, fontSize: 16, color: colors.ink }}>GPT-2 Transformer Architecture</h2>
        {loaded && (
          <span style={{ fontSize: 11, color: colors.inkMuted48 }}>
            {arch.model_name} · {nLayers} layers · {arch.n_heads} heads
          </span>
        )}
        <div style={{ flex: 1 }} />
        <input
          className="input-text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void runPrompt(); }}
          placeholder="Enter prompt to populate activations..."
          style={{ flex: 1, minWidth: 280, maxWidth: 480 }}
        />
        <button className="btn btn-primary" onClick={() => void runPrompt()} disabled={promptRunning}>
          {promptRunning ? <>Running<Spinner /></> : 'Run Prompt'}
        </button>
        {promptStatus && <span className="hint" style={{ fontSize: 11 }}>{promptStatus}</span>}
      </div>

      {error && (
        <div className="card" style={{ marginBottom: 14, background: colors.dangerSoft, border: `1px solid ${colors.dangerBorder}` }}>
          <p style={{ margin: 0, color: colors.dangerText, fontSize: 13 }}>{error}</p>
        </div>
      )}

      {archLoading ? (
        <div className="card"><p className="hint">Loading architecture from live model…<Spinner /></p></div>
      ) : !loaded ? (
        <div className="card" style={{ textAlign: 'center', padding: 40 }}>
          <p style={{ color: colors.inkMuted48, fontSize: 13 }}>{arch?.error || 'Model not loaded.'}</p>
          <button className="btn btn-primary" onClick={() => void loadModel()} disabled={loadingModel}>
            {loadingModel ? <>Loading GPT-2<Spinner /></> : 'Load GPT-2'}
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 0, flex: 1, overflow: 'hidden' }}>
          <div style={{
            flex: 1, overflowX: 'auto', overflowY: 'hidden',
            paddingBottom: 14,
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6, padding: 4, minWidth: '0' }}>
              <div style={{
                display: 'flex', flexDirection: 'column', gap: 6,
                width: 100, flexShrink: 0, padding: 10, borderRadius: 8,
                background: colors.surface, border: `1px solid ${colors.hairline}`,
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: colors.inkMuted48, textAlign: 'center' }}>WTE</div>
                <div style={{ fontSize: 9, color: colors.bodyMuted }}>
                  {fmt(arch.modules?.find((m: any) => m.id === 'wte')?.n_params)} params
                </div>
                <div style={{ fontSize: 9, color: colors.bodyMuted }}>
                  dim: {arch.modules?.find((m: any) => m.id === 'wte')?.shape?.[1] ?? '—'}
                </div>
              </div>

              <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', color: colors.hairline }}>
                →
              </div>

              {layers.map((layer: any, idx: number) => (
                <BlockCard key={layer.label || idx} layer={layer} idx={idx} />
              ))}

              <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', color: colors.hairline }}>
                →
              </div>

              <div style={{
                display: 'flex', flexDirection: 'column', gap: 6,
                width: 100, flexShrink: 0, padding: 10, borderRadius: 8,
                background: colors.surface, border: `1px solid ${colors.hairline}`,
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: colors.inkMuted48, textAlign: 'center' }}>LN_f</div>
                <div style={{ fontSize: 9, color: colors.bodyMuted }}>
                  {fmt(arch.modules?.find((m: any) => m.id === 'ln_f')?.n_params)} params
                </div>
                <div style={{ fontSize: 9, color: colors.bodyMuted }}>
                  dim: {arch.modules?.find((m: any) => m.id === 'ln_f')?.dim ?? '—'}
                </div>
              </div>

              <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', color: colors.hairline }}>
                →
              </div>

              <div style={{
                display: 'flex', flexDirection: 'column', gap: 6,
                width: 100, flexShrink: 0, padding: 10, borderRadius: 8,
                background: colors.surface, border: `1px solid ${colors.hairline}`,
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: colors.inkMuted48, textAlign: 'center' }}>LM Head</div>
                <div style={{ fontSize: 9, color: colors.bodyMuted }}>
                  {fmt(arch.modules?.find((m: any) => m.id === 'lm_head')?.n_params)} params
                </div>
                <div style={{ fontSize: 9, color: colors.bodyMuted }}>
                  vocab: {arch.vocab_size ?? '—'}
                </div>
              </div>
            </div>
          </div>

          {/* Detail panel */}
          <div style={{ width: 380, flexShrink: 0, borderLeft: `1px solid ${colors.hairline}`, padding: 14, overflowY: 'auto', maxHeight: 'calc(100vh - 180px)' }}>
            {selectedNode === null ? (
              <div className="card" style={{ border: `1px dashed ${colors.hairline}`, textAlign: 'center', padding: 28 }}>
                <p className="hint" style={{ margin: 0 }}>
                  Click an attention head, MLP neuron, or component to inspect its weights, connections, and activations.
                </p>
              </div>
            ) : nodeLoading ? (
              <div className="card"><p className="hint">Loading node detail…<Spinner /></p></div>
            ) : nodeDetails ? (
              <NodeInspector details={nodeDetails} selectedNodeId={selectedNode} />
            ) : (
              <div className="card"><p className="hint">No detail available for this node.</p></div>
            )}
          </div>
        </div>
      )}

      {loaded && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11, color: colors.inkMuted48, marginTop: 8 }}>
          <span>Layers: {nLayers}</span>
          <span>Heads/Layer: {arch.n_heads}</span>
          <span>MLP Neurons: {arch.d_mlp}</span>
          <span>Parameters: {fmt(arch.n_params)}</span>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Node Inspector                                                     */
/* ------------------------------------------------------------------ */

function NodeInspector({ details, selectedNodeId }: { details: any; selectedNodeId: string }) {
  if (!details) return null;

  // Layer detail
  if (details.status === 'ok' && details.layer !== undefined && details.num_attention_heads !== undefined) {
    return (
      <div className="card" style={{ padding: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <h3 style={{ margin: 0, fontSize: 14, color: colors.purpleBorder }}>
            {details.path} <span style={{ color: colors.inkMuted48, fontSize: 11 }}>· {fmt(details.n_params)} params</span>
          </h3>
          <button className="btn btn-ghost btn-sm" onClick={() => {}}>✕</button>
        </div>
        <KV k="residual stream" v={`d_model ${details.residual_stream_dim}`} />
        <KV k="attention heads" v={`${details.num_attention_heads} heads`} />
        <KV k="MLP neurons" v={String(details.num_mlp_neurons)} />
        <KV k="has activations" v={details.has_activations ? 'yes' : 'no (run a prompt)'} />
        {details.activation_summary && (
          <KV k="fraction active (last tok)" v={`${Math.round(details.activation_summary.fraction_active_last * 100)}%`} />
        )}

        {/* Heads */}
        {details.attention_heads?.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: colors.bodyMuted, marginBottom: 6 }}>ATTENTION HEADS</div>
            {details.attention_heads.map((h: any) => (
              <div key={h.head_index} style={{ fontSize: 10, padding: '2px 0', color: colors.purpleBorder }}>
                H{h.head_index}: Q {h.q_weight_l2?.toFixed(3)} K {h.k_weight_l2?.toFixed(3)} V {h.v_weight_l2?.toFixed(3)} O {h.o_weight_l2?.toFixed(3)}
              </div>
            ))}
          </div>
        )}

        {/* MLP */}
        {details.mlp && (
          <div style={{ marginTop: 10, borderTop: `1px solid ${colors.hairline}`, paddingTop: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: colors.bodyMuted, marginBottom: 4 }}>MLP WEIGHTS</div>
            <KV k="c_fc" v={details.mlp.c_fc_shape ? `[${details.mlp.c_fc_shape.join(' × ')}]` : ''} mono />
            <KV k="c_proj" v={details.mlp.c_proj_shape ? `[${details.mlp.c_proj_shape.join(' × ')}]` : ''} mono />
            {details.mlp.in_weight_l2_stats && (
              <>
                <MiniBar label="w_in µ" value={details.mlp.in_weight_l2_stats.mean} max={details.mlp.in_weight_l2_stats.max || 1} />
                <MiniBar label="w_in σ" value={details.mlp.in_weight_l2_stats.std} max={details.mlp.in_weight_l2_stats.max || 1} color={colors.primary} />
              </>
            )}
          </div>
        )}

        {/* Top active neurons */}
        {details.top_active_neurons?.length > 0 && (
          <div style={{ marginTop: 10, borderTop: `1px solid ${colors.hairline}`, paddingTop: 8 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, marginBottom: 4 }}>TOP ACTIVE NEURONS</div>
            {details.top_active_neurons.slice(0, 8).map((n: any) => (
              <div key={n.neuron_index} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, padding: '1px 0' }}>
                <span style={{ color: colors.purpleBorder }}>N{n.neuron_index}</span>
                <span style={{ fontFamily: 'monospace', color: colors.bodyMuted }}>{n.activation?.toFixed(4)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Neuron detail
  if (details.status === 'ok' && details.id && details.id.startsWith('L')) {
    return (
      <div className="card" style={{ padding: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <h3 style={{ margin: 0, fontSize: 14, color: colors.purpleBorder }}>{details.id}</h3>
          <button className="btn btn-ghost btn-sm" onClick={() => {}}>✕</button>
        </div>
        <KV k="path" v={details.path} mono />
        <KV k="component" v={details.component} />
        <KV k="bias" v={details.bias?.toFixed(4) ?? '—'} mono />
        <KV k="in weight L2" v={details.in_weight_l2?.toFixed(3)} mono />
        <KV k="out weight L2" v={details.out_weight_l2?.toFixed(3)} mono />

        {details.top_input_weights_positive?.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, color: colors.inkMuted48, marginBottom: 3 }}>top input weights (+)</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {details.top_input_weights_positive.slice(0, 10).map((w: any, i: number) => (
                <span key={i} style={{ fontSize: 9, fontFamily: 'monospace', color: colors.primary, background: colors.canvas, padding: '1px 4px', borderRadius: 3 }}>
                  d{w.dim}: +{w.weight.toFixed(2)}
                </span>
              ))}
            </div>
          </div>
        )}

        {details.top_output_weights_positive?.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, color: colors.inkMuted48, marginBottom: 3 }}>top output weights (+)</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {details.top_output_weights_positive.slice(0, 10).map((w: any, i: number) => (
                <span key={i} style={{ fontSize: 9, fontFamily: 'monospace', color: colors.primary, background: colors.canvas, padding: '1px 4px', borderRadius: 3 }}>
                  d{w.dim}: +{w.weight.toFixed(2)}
                </span>
              ))}
            </div>
          </div>
        )}

        {details.per_token_activations?.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, color: colors.inkMuted48, marginBottom: 3 }}>per-token activations</div>
            <div style={{ display: 'flex', gap: 1, alignItems: 'flex-end', height: 30, overflowX: 'auto' }}>
              {details.per_token_activations.slice(0, 40).map((t: any, i: number) => {
                const max = Math.max(...details.per_token_activations.map((x: any) => Math.abs(x.activation)), 0.01);
                const pct = Math.max(4, (Math.abs(t.activation) / max) * 100);
                return (
                  <div key={i} style={{ width: 4, height: `${pct}%`, background: t.activation >= 0 ? colors.primary : colors.danger, borderRadius: 1 }} title={`${t.token}: ${t.activation.toFixed(4)}`} />
                );
              })}
            </div>
          </div>
        )}

        {details.description && (
          <p className="hint" style={{ fontSize: 10, marginTop: 8 }}>{details.description}</p>
        )}
      </div>
    );
  }

  // Attention head detail
  if (details.status === 'ok' && details.matrix !== undefined) {
    return (
      <div className="card" style={{ padding: 14 }}>
        <h3 style={{ margin: 0, fontSize: 14, color: colors.purpleBorder, marginBottom: 8 }}>Attention Pattern</h3>
        {details.matrix?.length > 0 && details.str_tokens?.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 8 }}>
              <thead>
                <tr><th style={{ width: 24 }} />{details.str_tokens.map((t: string, i: number) => (
                  <th key={i} style={{ width: 24, textAlign: 'center', fontWeight: 500, padding: '1px 0', maxWidth: 24, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: colors.inkMuted48 }}>{t.slice(0, 3)}</th>
                ))}</tr>
              </thead>
              <tbody>
                {details.matrix.map((row: number[], qi: number) => (
                  <tr key={qi}>
                    <td style={{ color: colors.inkMuted48, paddingRight: 2, maxWidth: 36, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{(details.str_tokens[qi] || '').slice(0, 4)}</td>
                    {row.map((v: number, ki: number) => (
                      <td key={ki} style={{ width: 24, height: 24, background: ramp(Math.max(0, v), [hexToRgb(colors.surfaceBlack), hexToRgb(colors.primaryOnDark), hexToRgb(colors.warning)]), border: `1px solid ${rgba(colors.bodyMuted, 0.08)}`, borderRadius: 1, textAlign: 'center', color: colors.ink, fontSize: 7 }}>
                        {v.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="hint" style={{ fontSize: 11 }}>Run a prompt to see attention patterns.</p>
        )}
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: 14 }}>
      <h3 style={{ margin: 0, fontSize: 14, color: colors.purpleBorder, marginBottom: 8 }}>Node Detail</h3>
      <pre style={{ fontSize: 10, color: colors.bodyMuted, whiteSpace: 'pre-wrap', maxHeight: 300, overflowY: 'auto' }}>
        {JSON.stringify(details, null, 2).slice(0, 3000)}
      </pre>
    </div>
  );
}
