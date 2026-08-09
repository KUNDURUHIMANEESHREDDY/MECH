import React, { useEffect, useMemo, useState } from 'react';
import { Share2, Play, Loader2, AlertTriangle } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useModel } from '../../shared/hooks/useModel';
import type { FC, PanelContext } from '../../shared/types';

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

interface Edge {
  from: number;
  to: number;
  weight: number;
}

/**
 * Knowledge Graph: token-level attention graph.
 * Nodes are tokens; edges aggregate attention across all heads/layers,
 * drawn as a circular layout with edge weight as opacity. Click a token to
 * highlight its incoming/outgoing edges.
 */
export const KnowledgeGraphPanel: FC<PanelContext> = () => {
  const { state: model, load, infer, clearError, listModels } = useModel();
  const [prompt, setPrompt] = useState('The cat chased the dog. The cat chased the');
  const [hovered, setHovered] = useState<number | null>(null);
  const [edgeCount, setEdgeCount] = useState(24);
  const [stats, setStats] = useState<{
    nodes: number;
    edges: number;
    meanWeight: number;
    maxWeight: number;
  } | null>(null);

  useEffect(() => {
    if (model.availableModels.length === 0 && !model.loading) void listModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const tokens = model.result?.tokens.map((t) => t.text) ?? [];
  const layers = model.result?.layers ?? [];

  const nodes = useMemo(() => tokens.map((text, i) => ({ id: i, text })), [tokens]);

  const edges = useMemo<Edge[]>(() => {
    if (!layers.length) return [];
    const agg: Map<string, number> = new Map();
    for (const layer of layers) {
      for (const head of layer.heads) {
        const m = head.attentionMatrix;
        for (let i = 0; i < m.length; i++) {
          for (let j = 0; j < m[i].length; j++) {
            const key = `${i}:${j}`;
            agg.set(key, (agg.get(key) ?? 0) + m[i][j]);
          }
        }
      }
    }
    const totalHeads = layers.length > 0 ? layers[0].heads.length * layers.length : 1;
    const list: Edge[] = [];
    agg.forEach((w, key) => {
      const [from, to] = key.split(':').map(Number);
      list.push({ from, to, weight: w / totalHeads });
    });
    list.sort((a, b) => b.weight - a.weight);
    return list;
  }, [layers]);

  const visibleEdges = useMemo(() => edges.slice(0, Math.max(6, edgeCount)), [edges, edgeCount]);

  useEffect(() => {
    if (nodes.length && edges.length) {
      const mean = edges.reduce((a, e) => a + e.weight, 0) / edges.length;
      setStats({ nodes: nodes.length, edges: edges.length, meanWeight: mean, maxWeight: edges[0].weight });
    } else {
      setStats(null);
    }
  }, [nodes.length, edges, tokens.length]);

  // Circular layout
  const R = 150;
  const CX = 200;
  const CY = 200;
  const pos = useMemo(() => {
    return nodes.map((n, i) => {
      const angle = (i / Math.max(nodes.length, 1)) * 2 * Math.PI - Math.PI / 2;
      return { x: CX + R * Math.cos(angle), y: CY + R * Math.sin(angle), id: n.id, text: n.text };
    });
  }, [nodes]);

  const maxW = visibleEdges.length ? visibleEdges[0].weight : 1;
  const connectedOf = useMemo(() => {
    const set: Set<number> = new Set();
    for (const e of visibleEdges) {
      if (e.from === hovered || e.to === hovered) {
        set.add(e.from);
        set.add(e.to);
      }
    }
    return set;
  }, [visibleEdges, hovered]);

  const handleRun = async () => {
    clearError();
    await infer(prompt);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>
      {!model.loaded && (
        <div style={card}>
          <div style={{ fontWeight: 700, color: colors.ink }}>Model</div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <select value={model.availableModels[0] ?? 'gpt2'} onChange={(e) => load(e.target.value)} style={{ padding: '5px 8px', borderRadius: 6, border: `1px solid ${colors.hairline}`, background: colors.canvas, fontSize: 12, color: colors.ink }}>
              {(model.availableModels.length ? model.availableModels : ['gpt2']).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <button onClick={() => load(model.availableModels[0] ?? 'gpt2')} disabled={model.loading} style={{ ...btn, opacity: model.loading ? 0.5 : 1 }}>
              {model.loading ? <Loader2 size={13} /> : null} Load Model
            </button>
          </div>
        </div>
      )}

      {!model.result && (
        <div style={card}>
          <div style={{ fontWeight: 700, color: colors.ink }}>Run inference to build the graph</div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, background: colors.canvas, color: colors.ink }}
          />
          <div>
            <button onClick={handleRun} disabled={!model.loaded || model.running} style={{ ...btn, opacity: !model.loaded || model.running ? 0.5 : 1 }}>
              {model.running ? <Loader2 size={13} /> : <Play size={13} />} Run
            </button>
          </div>
        </div>
      )}

      {model.error && (
        <div style={{ background: colors.dangerSoft, color: colors.dangerText, border: `1px solid ${colors.dangerBorder}`, borderRadius: 8, padding: '8px 12px', display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 }}>
          <AlertTriangle size={14} /> {model.error}
        </div>
      )}

      {tokens.length > 0 && (
        <>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', gap: 6, alignItems: 'center', fontSize: 12, color: colors.bodyMuted }}>
              Edges shown
              <input
                type="range"
                min={6}
                max={Math.max(60, edges.length)}
                value={edgeCount}
                onChange={(e) => setEdgeCount(Number(e.target.value))}
                style={{ width: 140, accentColor: colors.primary }}
              />
              <span style={{ width: 34, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{edgeCount}</span>
            </label>
            {stats && (
              <span style={{ fontSize: 11, color: colors.bodyMuted }}>
                {stats.nodes} tokens · {stats.edges} pairs · avg {stats.meanWeight.toFixed(3)}
              </span>
            )}
          </div>

          <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, overflow: 'hidden', background: colors.canvas }}>
            <svg width={400} height={400} viewBox="0 0 400 400" style={{ display: 'block', width: '100%' }}>
              {visibleEdges.map((e, i) => {
                const a = pos[e.from];
                const b = pos[e.to];
                if (!a || !b) return null;
                const opacity = 0.12 + (e.weight / maxW) * 0.78;
                const active = hovered !== null && (e.from === hovered || e.to === hovered);
                return (
                  <line
                    key={i}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={active ? colors.primary : colors.bodyMuted}
                    strokeWidth={active ? 2 : 0.8}
                    strokeOpacity={active ? 0.95 : opacity}
                  />
                );
              })}
              {pos.map((p) => {
                const active = hovered !== null && (p.id === hovered || connectedOf.has(p.id));
                return (
                  <g key={p.id} onMouseEnter={() => setHovered(p.id)} onMouseLeave={() => setHovered(null)} style={{ cursor: 'pointer' }}>
                    <circle cx={p.x} cy={p.y} r={hovered === p.id ? 14 : 10} fill={active || hovered === p.id ? colors.primary : colors.surfacePearl} stroke={colors.primary} strokeWidth={hovered === p.id ? 2 : 1} />
                    <text
                      x={p.x}
                      y={p.y + 3}
                      textAnchor="middle"
                      fill={hovered === p.id ? colors.onPrimary : colors.ink}
                      fontSize={8}
                      fontWeight={hovered === p.id ? 700 : 500}
                      pointerEvents="none"
                    >
                      {p.text.slice(0, 14)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          <div style={{ fontSize: 11, color: colors.bodyMuted, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Share2 size={12} color={colors.primary} />
            Hover a token to highlight its top outgoing/incoming attention edges.
          </div>
        </>
      )}
    </div>
  );
};