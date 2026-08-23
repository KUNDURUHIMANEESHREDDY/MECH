import React, { useState } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { useSelectionStore } from '../../shared/stores/selection';
import { colors } from '../../design/tokens/colors';
import { Network, ShieldCheck, ArrowRight, Brain, FlaskConical, Layers, Filter } from 'lucide-react';

export const EvidenceGraphView: React.FC = () => {
  const {
    activeInvestigation,
    hypotheses,
    runs,
    evidence,
    selectComponent,
  } = useResearchStore();

  const selection = useSelectionStore();
  const [filterType, setFilterType] = useState<string>('ALL');
  const [selectedNode, setSelectedNode] = useState<any | null>(null);

  // Compile unified graph nodes and edges from real database entities
  const nodes: any[] = [];
  const edges: any[] = [];

  // Investigation root node
  if (activeInvestigation) {
    nodes.push({
      id: activeInvestigation.id,
      label: activeInvestigation.title,
      type: 'INVESTIGATION',
      sourceType: 'USER_DEFINED',
      details: activeInvestigation.research_question,
    });
  }

  // Hypothesis nodes
  hypotheses.forEach((h) => {
    nodes.push({
      id: h.id,
      label: h.title,
      type: 'HYPOTHESIS',
      sourceType: 'USER_DEFINED',
      status: h.status,
      details: h.statement,
      component: h.target_component,
    });
    if (activeInvestigation) {
      edges.push({
        id: `e_${activeInvestigation.id}_${h.id}`,
        from: activeInvestigation.id,
        to: h.id,
        relation: 'investigates',
      });
    }
  });

  // Experiment Run nodes
  runs.forEach((r) => {
    nodes.push({
      id: r.id,
      label: `Run ${r.id.slice(-6)} (ΔL=${r.delta_logit.toFixed(2)})`,
      type: 'INTERVENTION_RUN',
      sourceType: 'COMPUTED',
      details: `Clean Logit: ${r.baseline_logit.toFixed(2)} → Intervened: ${r.intervened_logit.toFixed(2)}`,
      delta_logit: r.delta_logit,
    });
    if (r.investigation_id) {
      edges.push({
        id: `e_${r.investigation_id}_${r.id}`,
        from: r.investigation_id,
        to: r.id,
        relation: 'executed_in',
      });
    }
  });

  // Evidence Records
  evidence.forEach((e) => {
    nodes.push({
      id: e.id,
      label: e.claim,
      type: 'EVIDENCE',
      sourceType: e.source_type || 'COMPUTED',
      level: e.evidence_level,
      supports: e.supports_hypothesis,
      details: `Metric: ${e.metric_name}=${e.metric_value.toFixed(2)}`,
    });
    if (e.hypothesis_id) {
      edges.push({
        id: `e_${e.id}_${e.hypothesis_id}`,
        from: e.id,
        to: e.hypothesis_id,
        relation: e.supports_hypothesis ? 'supports' : 'contradicts',
      });
    }
    if (e.experiment_run_id) {
      edges.push({
        id: `e_${e.experiment_run_id}_${e.id}`,
        from: e.experiment_run_id,
        to: e.id,
        relation: 'derived_from',
      });
    }
  });

  const getSourceTypeBadge = (source: string) => {
    switch (source) {
      case 'COMPUTED':
        return <span style={{ fontSize: 9, fontWeight: 800, padding: '1px 5px', borderRadius: 4, backgroundColor: colors.successSoft, color: colors.successText, border: `1px solid ${colors.successBorder}` }}>COMPUTED</span>;
      case 'OBSERVED':
        return <span style={{ fontSize: 9, fontWeight: 800, padding: '1px 5px', borderRadius: 4, backgroundColor: colors.primarySoft, color: colors.primary, border: `1px solid ${colors.border}` }}>OBSERVED</span>;
      case 'AI_GENERATED':
        return <span style={{ fontSize: 9, fontWeight: 800, padding: '1px 5px', borderRadius: 4, backgroundColor: colors.purpleSoft, color: colors.purpleText, border: `1px solid ${colors.border}` }}>AI_INTERPRETATION</span>;
      default:
        return <span style={{ fontSize: 9, fontWeight: 800, padding: '1px 5px', borderRadius: 4, backgroundColor: colors.surfacePearl, color: colors.bodyMuted, border: `1px solid ${colors.border}` }}>{source}</span>;
    }
  };

  const handleNodeClick = (node: any) => {
    setSelectedNode(node);
    if (node.component) {
      const comp = node.component;
      if (comp.startsWith('L') && comp.includes('H')) {
        const parts = comp.split('H');
        const layer = parseInt(parts[0].replace('L', ''), 10);
        const head = parseInt(parts[1], 10);
        selection.setSelectedHead(layer, head);
        selectComponent({ name: comp, layer, head, componentType: 'head' });
      }
    }
  };

  const filteredNodes = nodes.filter((n) => {
    if (filterType === 'ALL') return true;
    return n.type === filterType;
  });

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            Scientific Provenance Graph
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Evidence & Knowledge Architecture
          </h2>
        </div>

        {/* Filter Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Filter size={14} style={{ color: colors.bodyMuted }} />
          {['ALL', 'HYPOTHESIS', 'INTERVENTION_RUN', 'EVIDENCE'].map((t) => (
            <button
              key={t}
              onClick={() => setFilterType(t)}
              style={{
                padding: '4px 10px',
                borderRadius: 999,
                border: `1px solid ${filterType === t ? colors.primary : colors.border}`,
                backgroundColor: filterType === t ? colors.primary : colors.surfaceTile1,
                color: filterType === t ? colors.onPrimary : colors.body,
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              {t.replace('_', ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Graph Visualizer & Inspector Split */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedNode ? '2fr 1fr' : '1fr', gap: 16 }}>
        {/* Node Grid View */}
        <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 0.4 }}>
            Knowledge Nodes ({filteredNodes.length})
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 10 }}>
            {filteredNodes.map((n) => (
              <div
                key={n.id}
                onClick={() => handleNodeClick(n)}
                style={{
                  border: `1px solid ${selectedNode?.id === n.id ? colors.primary : colors.border}`,
                  borderRadius: 8,
                  padding: 12,
                  backgroundColor: colors.canvas,
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                  boxShadow: selectedNode?.id === n.id ? `0 0 0 2px ${colors.primary}` : 'none',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase' }}>
                    {n.type}
                  </span>
                  {getSourceTypeBadge(n.sourceType)}
                </div>

                <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, lineHeight: 1.3 }}>
                  {n.label}
                </div>

                <div style={{ fontSize: 11, color: colors.bodyMuted, lineHeight: 1.4 }}>
                  {n.details}
                </div>

                {n.component && (
                  <div style={{ marginTop: 2 }}>
                    <code style={{ fontSize: 10, backgroundColor: colors.surfacePearl, padding: '2px 5px', borderRadius: 4 }}>
                      Component: {n.component}
                    </code>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Relations / Edges List */}
          <div style={{ marginTop: 20, borderTop: `1px solid ${colors.borderLight}`, paddingTop: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase', marginBottom: 8 }}>
              Evidence Links ({edges.length})
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {edges.map((e) => (
                <div
                  key={e.id}
                  style={{
                    fontSize: 11,
                    backgroundColor: colors.canvas,
                    border: `1px solid ${colors.border}`,
                    borderRadius: 6,
                    padding: '4px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  <span style={{ color: colors.body }}>{e.from.slice(0, 8)}…</span>
                  <span style={{ fontWeight: 700, color: e.relation === 'supports' ? colors.successText : (e.relation === 'contradicts' ? colors.dangerText : colors.primary) }}>
                    → {e.relation} →
                  </span>
                  <span style={{ color: colors.body }}>{e.to.slice(0, 8)}…</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Selected Node Inspector */}
        {selectedNode && (
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: colors.ink, textTransform: 'uppercase' }}>Node Inspector</span>
              <button
                onClick={() => setSelectedNode(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 16, color: colors.bodyMuted }}
              >
                ×
              </button>
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: colors.ink, marginBottom: 8 }}>
              {selectedNode.label}
            </div>

            <div style={{ marginBottom: 10 }}>
              {getSourceTypeBadge(selectedNode.sourceType)}
            </div>

            <div style={{ fontSize: 12, color: colors.body, marginBottom: 12, lineHeight: 1.45, backgroundColor: colors.canvas, padding: 10, borderRadius: 6, border: `1px solid ${colors.border}` }}>
              {selectedNode.details}
            </div>

            <div style={{ fontSize: 11, color: colors.bodyMuted, display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div>Node ID: <code>{selectedNode.id}</code></div>
              <div>Node Type: <b>{selectedNode.type}</b></div>
              {selectedNode.status && <div>Hypothesis Status: <b>{selectedNode.status}</b></div>}
              {selectedNode.component && <div>Associated Component: <b>{selectedNode.component}</b></div>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
