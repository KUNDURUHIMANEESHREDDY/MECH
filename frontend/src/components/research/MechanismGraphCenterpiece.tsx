import React, { useState } from 'react';
import {
  Layers,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  Info,
  Sliders,
  ExternalLink,
  Target,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface CircuitNode {
  id: string;
  label: string;
  type: 'ATTENTION_HEAD' | 'MLP' | 'RESIDUAL' | 'LOGITS';
  layer: number;
  epistemicStatus: 'ESTABLISHED' | 'SUPPORTED' | 'CANDIDATE' | 'UNKNOWN';
  x: number;
  y: number;
}

interface CircuitEdge {
  id: string;
  source: string;
  target: string;
  relationship: 'CAUSAL' | 'OBSERVATIONAL' | 'CANDIDATE';
  evidenceId: string;
  interventions: number;
  controls: number;
  replications: number;
  deltaLogit: number;
  status: string;
  limitations: string;
}

interface Props {
  onSelectEdge?: (edge: CircuitEdge) => void;
  onSelectNode?: (node: CircuitNode) => void;
}

export const MechanismGraphCenterpiece: React.FC<Props> = ({ onSelectEdge, onSelectNode }) => {
  const [selectedEdge, setSelectedEdge] = useState<CircuitEdge | null>(null);
  const [selectedNode, setSelectedNode] = useState<CircuitNode | null>(null);

  const nodes: CircuitNode[] = [
    { id: 'L8H1', label: 'L8H1 (Duplicate Token)', type: 'ATTENTION_HEAD', layer: 8, epistemicStatus: 'SUPPORTED', x: 60, y: 80 },
    { id: 'L9H9', label: 'L9H9 (Name Mover)', type: 'ATTENTION_HEAD', layer: 9, epistemicStatus: 'SUPPORTED', x: 260, y: 60 },
    { id: 'MLP_L8', label: 'MLP L8 (Feedforward)', type: 'MLP', layer: 8, epistemicStatus: 'CANDIDATE', x: 260, y: 170 },
    { id: 'RESIDUAL', label: 'Residual Stream (L10)', type: 'RESIDUAL', layer: 10, epistemicStatus: 'ESTABLISHED', x: 460, y: 110 },
    { id: 'LOGITS', label: 'Target Logit (Mary)', type: 'LOGITS', layer: 12, epistemicStatus: 'ESTABLISHED', x: 640, y: 110 },
  ];

  const edges: CircuitEdge[] = [
    {
      id: 'e1',
      source: 'L8H1',
      target: 'L9H9',
      relationship: 'CAUSAL',
      evidenceId: 'EVID-17',
      interventions: 3,
      controls: 2,
      replications: 3,
      deltaLogit: 2.13,
      status: 'CAUSAL MATCH',
      limitations: 'Measured on 3-prompt subset; full 100-prompt distribution required for complete circuit closure.',
    },
    {
      id: 'e2',
      source: 'L9H9',
      target: 'RESIDUAL',
      relationship: 'CAUSAL',
      evidenceId: 'EVID-42',
      interventions: 3,
      controls: 2,
      replications: 3,
      deltaLogit: 1.85,
      status: 'CAUSAL MATCH',
      limitations: 'Direct logit write verified; backup name mover compensation may partially buffer single-head knockout.',
    },
    {
      id: 'e3',
      source: 'MLP_L8',
      target: 'RESIDUAL',
      relationship: 'CANDIDATE',
      evidenceId: 'EVID-89',
      interventions: 1,
      controls: 0,
      replications: 1,
      deltaLogit: 0.42,
      status: 'CANDIDATE (Unresolved)',
      limitations: 'Correlation observed without negative control baseline; requires isolated ablation.',
    },
    {
      id: 'e4',
      source: 'RESIDUAL',
      target: 'LOGITS',
      relationship: 'CAUSAL',
      evidenceId: 'EVID-90',
      interventions: 5,
      controls: 3,
      replications: 5,
      deltaLogit: 2.45,
      status: 'ESTABLISHED',
      limitations: 'Direct unembedding projection.',
    },
  ];

  const handleNodeClick = (node: CircuitNode) => {
    setSelectedNode(node);
    setSelectedEdge(null);
    if (onSelectNode) onSelectNode(node);
  };

  const handleEdgeClick = (edge: CircuitEdge) => {
    setSelectedEdge(edge);
    setSelectedNode(null);
    if (onSelectEdge) onSelectEdge(edge);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: colors.canvas,
        color: colors.ink,
        overflow: 'hidden',
      }}
    >
      {/* Canvas Header */}
      <div
        style={{
          padding: '12px 18px',
          borderBottom: `1px solid ${colors.hairline}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: colors.surfaceTile1,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Layers size={16} color={colors.primary} />
          <div>
            <h3 style={{ margin: 0, fontSize: 13, fontWeight: 700 }}>
              Mechanistic Circuit Graph (v2 Active Lineage)
            </h3>
            <div style={{ fontSize: 11, color: colors.bodyMuted }}>
              Interactive structural and causal pathway topology. Click any node or edge to inspect evidence provenance.
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: colors.successText }} /> Causal
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: colors.primary }} /> Observational
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: colors.bodyMuted }} /> Candidate
          </span>
        </div>
      </div>

      {/* SVG Canvas Area */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden', minHeight: 280, backgroundColor: colors.surfaceTile2 }}>
        <svg style={{ width: '100%', height: '100%', minHeight: 280 }}>
          <defs>
            <marker id="arrow-causal" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M 0 0 L 8 4 L 0 8 z" fill={colors.successText} />
            </marker>
            <marker id="arrow-cand" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M 0 0 L 8 4 L 0 8 z" fill={colors.bodyMuted} />
            </marker>
          </defs>

          {/* Render Edges */}
          {edges.map((e) => {
            const src = nodes.find((n) => n.id === e.source)!;
            const tgt = nodes.find((n) => n.id === e.target)!;
            const isSelected = selectedEdge?.id === e.id;
            const strokeColor = e.relationship === 'CAUSAL' ? colors.successText : colors.bodyMuted;
            const strokeDash = e.relationship === 'CANDIDATE' ? '4,4' : 'none';

            return (
              <g key={e.id} onClick={() => handleEdgeClick(e)} style={{ cursor: 'pointer' }}>
                <line
                  x1={src.x + 80}
                  y1={src.y + 20}
                  x2={tgt.x}
                  y2={tgt.y + 20}
                  stroke={isSelected ? colors.primary : strokeColor}
                  strokeWidth={isSelected ? 3 : 2}
                  strokeDasharray={strokeDash}
                  markerEnd={e.relationship === 'CAUSAL' ? 'url(#arrow-causal)' : 'url(#arrow-cand)'}
                />
                {/* Edge Label Badge */}
                <rect
                  x={(src.x + tgt.x) / 2 + 10}
                  y={(src.y + tgt.y) / 2 + 5}
                  width={56}
                  height={16}
                  rx={3}
                  fill={colors.surfaceTile1}
                  stroke={isSelected ? colors.primary : colors.hairline}
                />
                <text
                  x={(src.x + tgt.x) / 2 + 38}
                  y={(src.y + tgt.y) / 2 + 16}
                  fill={colors.ink}
                  fontSize="9"
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  +{e.deltaLogit}
                </text>
              </g>
            );
          })}

          {/* Render Nodes */}
          {nodes.map((n) => {
            const isSelected = selectedNode?.id === n.id;
            return (
              <g key={n.id} transform={`translate(${n.x}, ${n.y})`} onClick={() => handleNodeClick(n)} style={{ cursor: 'pointer' }}>
                <rect
                  width={140}
                  height={40}
                  rx={6}
                  fill={colors.surfaceTile1}
                  stroke={isSelected ? colors.primary : colors.hairline}
                  strokeWidth={isSelected ? 2 : 1}
                />
                <circle cx={14} cy={20} r={5} fill={n.epistemicStatus === 'SUPPORTED' ? colors.successText : colors.primary} />
                <text x={26} y={18} fill={colors.ink} fontSize="11" fontWeight="700">
                  {n.id}
                </text>
                <text x={26} y={30} fill={colors.bodyMuted} fontSize="9">
                  {n.label.split('(')[1]?.replace(')', '') || n.type}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Detail Inspector Footer */}
      {(selectedEdge || selectedNode) && (
        <div
          style={{
            padding: 14,
            borderTop: `1px solid ${colors.hairline}`,
            backgroundColor: colors.surfaceTile1,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: 12,
          }}
        >
          {selectedEdge ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ fontWeight: 700, color: colors.ink }}>
                  Edge Inspector: <span style={{ fontFamily: 'monospace' }}>{selectedEdge.source} ➔ {selectedEdge.target}</span> ({selectedEdge.relationship})
                </div>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 4,
                    backgroundColor: colors.successSoft,
                    color: colors.successText,
                  }}
                >
                  {selectedEdge.status}
                </span>
              </div>
              <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                <strong>Evidence:</strong> {selectedEdge.evidenceId} · <strong>Interventions:</strong> {selectedEdge.interventions} · <strong>Negative Controls:</strong> {selectedEdge.controls} · <strong>Replications:</strong> {selectedEdge.replications}
              </div>
              <div style={{ fontSize: 10, color: colors.bodyMuted, fontStyle: 'italic' }}>
                <strong>Limitations:</strong> {selectedEdge.limitations}
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, width: '100%' }}>
              <div style={{ fontWeight: 700, color: colors.ink }}>
                Component Inspector: {selectedNode?.label}
              </div>
              <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                <strong>Layer:</strong> {selectedNode?.layer} · <strong>Status:</strong> {selectedNode?.epistemicStatus}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
