import React, { useState, useCallback, useEffect, useRef } from 'react';
import { api } from '../services/api';

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
  darkMode?: boolean;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const COLORS = {
  bg: '#0f0f1a',
  nodeEmbedding: '#60a5fa',
  nodeLayernorm: '#a78bfa',
  nodeAttention: '#facc15',
  nodeMlp: '#34d399',
  nodeOutput: '#f472b6',
  nodeBlockInput: '#60a5fa',
  nodeBlockOutput: '#60a5fa',
  edgeFeedforward: '#3a3a5a',
  edgeResidual: '#f97316',
  edgeAttention: '#818cf8',
  edgeHighlight: '#d0c0ff',
  text: '#e2e8f0',
  textDim: '#64748b',
  accent: '#d0c0ff',
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
};

const VIRIDIS = COLORS.viridis;
const BLOCK_W = 180;
const NODE_R = 6;
const LAYER_GAP = 160;
const NODE_GAP = 18;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/*
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
      <span style={{ color: '#888', minWidth: 140, flexShrink: 0 }}>{k}</span>
      <span style={mono ? { fontFamily: 'monospace' } : {}}>{v ?? '—'}</span>
    </div>
  );
}

function MiniBar({ label, value, max, color = '#d0c0ff' }: { label: string; value: number; max: number; color?: string }) {
  const pct = max > 0 ? Math.max(2, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, marginBottom: 2 }}>
      <span style={{ width: 34, color: '#888', flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, background: '#1e1e2e', borderRadius: 2, height: 6, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2 }} />
      </div>
      <span style={{ width: 44, textAlign: 'right', fontFamily: 'monospace', color: '#aaa', flexShrink: 0 }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */

export default function TransformerVisualizer({ onNavigate, darkMode }: Props) {
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
  const graph = graphData;

  return (
    <div data-testid="transformer-visualizer" style={{ padding: 16, height: '100%', display: 'flex', flexDirection: 'column', background: COLORS.bg }}>
      {/* Header + prompt bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
        <h2 style={{ margin: 0, fontSize: 16, color: '#eee' }}>GPT-2 Computational Graph</h2>
        {loaded && graph && (
          <span style={{ fontSize: 11, color: '#888' }}>
            {arch.model_name} · {graph.nodes.length} nodes · {graph.edges.length} edges
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
        <div className="card" style={{ marginBottom: 14, background: '#fef2f2', border: '1px solid #fecaca' }}>
          <p style={{ margin: 0, color: '#991b1b', fontSize: 13 }}>{error}</p>
        </div>
      )}

      {archLoading ? (
        <div className="card"><p className="hint">Loading architecture from live model…<Spinner /></p></div>
      ) : !loaded ? (
        <div className="card" style={{ textAlign: 'center', padding: 40 }}>
          <p style={{ color: '#bbb', fontSize: 13 }}>{arch?.error || 'Model not loaded.'}</p>
          <button className="btn btn-primary" onClick={() => void loadModel()} disabled={loadingModel}>
            {loadingModel ? <>Loading GPT-2<Spinner /></> : 'Load GPT-2'}
          </button>
        </div>
      ) : !graph ? (
        <div className="card"><p className="hint">Building graph…</p></div>
      ) : (
        <div style={{ display: 'flex', gap: 0, flex: 1, overflow: 'hidden' }}>
          {/* Graph SVG */}
          <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
            <svg
              ref={svgRef}
              width="100%"
              height="100%"
              viewBox={`0 0 ${1200} ${800}`}
              style={{ cursor: dragging ? 'grabbing' : 'grab', background: COLORS.bg }}
              onMouseDown={(e) => { setDragging(true); setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y }); }}
              onMouseMove={(e) => { if (dragging) setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y }); }}
              onMouseUp={() => setDragging(false)}
              onMouseLeave={() => setDragging(false)}
              onWheel={(e) => { e.preventDefault(); setZoom((z) => Math.max(0.2, Math.min(3, z * (e.deltaY > 0 ? 0.95 : 1.05)))); }}
            >
              <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
                {/* Edges */}
                {graph.edges.map((edge, i) => {
                  const fromNode = graph.nodes.find((n) => n.id === edge.from);
                  const toNode = graph.nodes.find((n) => n.id === edge.to);
                  if (!fromNode || !toNode) return null;
                  const isHighlighted = hoveredNode && (edge.from === hoveredNode || edge.to === hoveredNode);
                  const isSelected = selectedNode && (edge.from === selectedNode || edge.to === selectedNode);
                  const color = edge.type === 'residual' ? COLORS.edgeResidual : edge.type === 'attention' ? COLORS.edgeAttention : COLORS.edgeFeedforward;
                  const opacity = isHighlighted || isSelected ? 0.9 : 0.3;
                  const width = (isHighlighted || isSelected) ? 2 : 1;
                  const mx = (fromNode.x + toNode.x) / 2;
                  const my = (fromNode.y + toNode.y) / 2;
                  const dx = toNode.x - fromNode.x;
                  const dy = toNode.y - fromNode.y;
                  const cx = mx - dy * 0.1;
                  const cy = my + dx * 0.1;
                  return (
                    <path
                      key={i}
                      d={`M ${fromNode.x} ${fromNode.y} Q ${cx} ${cy} ${toNode.x} ${toNode.y}`}
                      stroke={color}
                      strokeWidth={width}
                      fill="none"
                      opacity={opacity}
                      strokeDasharray={edge.type === 'residual' ? '4 3' : 'none'}
                    />
                  );
                })}
                {/* Nodes */}
                {graph.nodes.map((node) => {
                  const isSelected = selectedNode === node.id;
                  const isHovered = hoveredNode === node.id;
                  const color = nodeColor(node);
                  const r = node.type === 'embedding' || node.type === 'output' ? NODE_R * 2 : NODE_R;
                  return (
                    <g
                      key={node.id}
                      onMouseEnter={() => setHoveredNode(node.id)}
                      onMouseLeave={() => setHoveredNode(null)}
                      onClick={() => { setSelectedNode(node.id); void loadNodeDetail(node.id); }}
                      style={{ cursor: 'pointer' }}
                    >
                      <circle
                        cx={node.x}
                        cy={node.y}
                        r={r}
                        fill={color}
                        opacity={isSelected ? 1 : isHovered ? 0.9 : 0.7}
                        stroke={isSelected ? '#fff' : 'none'}
                        strokeWidth={isSelected ? 2 : 0}
                      />
                      {node.type === 'attention-head' && (
                        <text x={node.x} y={node.y - r - 4} textAnchor="middle" fill={COLORS.textDim} fontSize={8} fontFamily="inherit">
                          {node.label}
                        </text>
                      )}
                      {node.type === 'mlp-neuron' && (
                        <circle
                          cx={node.x}
                          cy={node.y}
                          r={r * 0.5}
                          fill={ramp(Math.abs(node.activation || 0), VIRIDIS)}
                          opacity={0.8}
                        />
                      )}
                    </g>
                  );
                })}
              </g>
            </svg>
            {/* Zoom controls */}
            <div style={{ position: 'absolute', bottom: 12, right: 12, display: 'flex', gap: 4 }}>
              <button className="btn btn-sm" onClick={() => setZoom((z) => Math.min(3, z * 1.2))}>+</button>
              <button className="btn btn-sm" onClick={() => setZoom((z) => Math.max(0.2, z / 1.2))}>−</button>
              <button className="btn btn-sm" onClick={() => { setZoom(0.6); setPan({ x: 400, y: 300 }); }}>reset</button>
            </div>
          </div>

          {/* Detail panel */}
          <div style={{ width: 380, flexShrink: 0, borderLeft: '1px solid #2a2a4a', padding: 14, overflowY: 'auto', maxHeight: 'calc(100vh - 180px)' }}>
            {selectedNode === null ? (
              <div className="card" style={{ border: '1px dashed #2a2a4a', textAlign: 'center', padding: 28 }}>
                <p className="hint" style={{ margin: 0 }}>
                  Click a neuron or component node to inspect its weights, connections, and activations.
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
          <h3 style={{ margin: 0, fontSize: 14, color: '#d0c0ff' }}>
            {details.path} <span style={{ color: '#888', fontSize: 11 }}>· {fmt(details.n_params)} params</span>
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
            <div style={{ fontSize: 11, fontWeight: 700, color: '#aaa', marginBottom: 6 }}>ATTENTION HEADS</div>
            {details.attention_heads.map((h: any) => (
              <div key={h.head_index} style={{ fontSize: 10, padding: '2px 0', color: '#d0c0ff' }}>
                H{h.head_index}: Q {h.q_weight_l2?.toFixed(3)} K {h.k_weight_l2?.toFixed(3)} V {h.v_weight_l2?.toFixed(3)} O {h.o_weight_l2?.toFixed(3)}
              </div>
            ))}
          </div>
        )}

        {/* MLP */}
        {details.mlp && (
          <div style={{ marginTop: 10, borderTop: '1px solid #2a2a4a', paddingTop: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#aaa', marginBottom: 4 }}>MLP WEIGHTS</div>
            <KV k="c_fc" v={details.mlp.c_fc_shape ? `[${details.mlp.c_fc_shape.join(' × ')}]` : ''} mono />
            <KV k="c_proj" v={details.mlp.c_proj_shape ? `[${details.mlp.c_proj_shape.join(' × ')}]` : ''} mono />
            {details.mlp.in_weight_l2_stats && (
              <>
                <MiniBar label="w_in µ" value={details.mlp.in_weight_l2_stats.mean} max={details.mlp.in_weight_l2_stats.max || 1} />
                <MiniBar label="w_in σ" value={details.mlp.in_weight_l2_stats.std} max={details.mlp.in_weight_l2_stats.max || 1} color="#60a5fa" />
              </>
            )}
          </div>
        )}

        {/* Top active neurons */}
        {details.top_active_neurons?.length > 0 && (
          <div style={{ marginTop: 10, borderTop: '1px solid #2a2a4a', paddingTop: 8 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: '#aaa', marginBottom: 4 }}>TOP ACTIVE NEURONS</div>
            {details.top_active_neurons.slice(0, 8).map((n: any) => (
              <div key={n.neuron_index} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, padding: '1px 0' }}>
                <span style={{ color: '#d0c0ff' }}>N{n.neuron_index}</span>
                <span style={{ fontFamily: 'monospace', color: '#aaa' }}>{n.activation?.toFixed(4)}</span>
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
          <h3 style={{ margin: 0, fontSize: 14, color: '#d0c0ff' }}>{details.id}</h3>
          <button className="btn btn-ghost btn-sm" onClick={() => {}}>✕</button>
        </div>
        <KV k="path" v={details.path} mono />
        <KV k="component" v={details.component} />
        <KV k="bias" v={details.bias?.toFixed(4) ?? '—'} mono />
        <KV k="in weight L2" v={details.in_weight_l2?.toFixed(3)} mono />
        <KV k="out weight L2" v={details.out_weight_l2?.toFixed(3)} mono />

        {details.top_input_weights_positive?.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, color: '#999', marginBottom: 3 }}>top input weights (+)</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {details.top_input_weights_positive.slice(0, 10).map((w: any, i: number) => (
                <span key={i} style={{ fontSize: 9, fontFamily: 'monospace', color: '#7dd3fc', background: '#1e1e2e', padding: '1px 4px', borderRadius: 3 }}>
                  d{w.dim}: +{w.weight.toFixed(2)}
                </span>
              ))}
            </div>
          </div>
        )}

        {details.top_output_weights_positive?.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, color: '#999', marginBottom: 3 }}>top output weights (+)</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
              {details.top_output_weights_positive.slice(0, 10).map((w: any, i: number) => (
                <span key={i} style={{ fontSize: 9, fontFamily: 'monospace', color: '#7dd3fc', background: '#1e1e2e', padding: '1px 4px', borderRadius: 3 }}>
                  d{w.dim}: +{w.weight.toFixed(2)}
                </span>
              ))}
            </div>
          </div>
        )}

        {details.per_token_activations?.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 10, color: '#999', marginBottom: 3 }}>per-token activations</div>
            <div style={{ display: 'flex', gap: 1, alignItems: 'flex-end', height: 30, overflowX: 'auto' }}>
              {details.per_token_activations.slice(0, 40).map((t: any, i: number) => {
                const max = Math.max(...details.per_token_activations.map((x: any) => Math.abs(x.activation)), 0.01);
                const pct = Math.max(4, (Math.abs(t.activation) / max) * 100);
                return (
                  <div key={i} style={{ width: 4, height: `${pct}%`, background: t.activation >= 0 ? '#3b82f6' : '#ef4444', borderRadius: 1 }} title={`${t.token}: ${t.activation.toFixed(4)}`} />
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
        <h3 style={{ margin: 0, fontSize: 14, color: '#d0c0ff', marginBottom: 8 }}>Attention Pattern</h3>
        {details.matrix?.length > 0 && details.str_tokens?.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 8 }}>
              <thead>
                <tr><th style={{ width: 24 }} />{details.str_tokens.map((t: string, i: number) => (
                  <th key={i} style={{ width: 24, textAlign: 'center', fontWeight: 500, padding: '1px 0', maxWidth: 24, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#999' }}>{t.slice(0, 3)}</th>
                ))}</tr>
              </thead>
              <tbody>
                {details.matrix.map((row: number[], qi: number) => (
                  <tr key={qi}>
                    <td style={{ color: '#999', paddingRight: 2, maxWidth: 36, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{(details.str_tokens[qi] || '').slice(0, 4)}</td>
                    {row.map((v: number, ki: number) => (
                      <td key={ki} style={{ width: 24, height: 24, background: ramp(Math.max(0, v), [[30, 41, 59], [14, 165, 233], [250, 204, 21]]), border: '1px solid rgba(148,163,184,0.08)', borderRadius: 1, textAlign: 'center', color: '#0f172a', fontSize: 7 }}>
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
      <h3 style={{ margin: 0, fontSize: 14, color: '#d0c0ff', marginBottom: 8 }}>Node Detail</h3>
      <pre style={{ fontSize: 10, color: '#aaa', whiteSpace: 'pre-wrap', maxHeight: 300, overflowY: 'auto' }}>
        {JSON.stringify(details, null, 2).slice(0, 3000)}
      </pre>
    </div>
  );
}
