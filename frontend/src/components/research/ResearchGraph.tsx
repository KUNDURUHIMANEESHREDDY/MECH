import React, { useState, useMemo } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import {
  Brain,
  FlaskConical,
  Target,
  Zap,
  Eye,
  ShieldCheck,
  Award,
  ArrowDown,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Filter,
  Search,
  RefreshCw,
  Layers,
  GitBranch,
} from 'lucide-react';

type NodeCategory = 'HYPOTHESIS' | 'EXPERIMENT' | 'COMPONENT' | 'INTERVENTION' | 'OBSERVATION' | 'EVIDENCE' | 'FINDING';

interface GraphNode {
  id: string;
  category: NodeCategory;
  label: string;
  subtitle?: string;
  status?: string;
  evidenceLevel?: string;
  supportsHypothesis?: boolean;
  deltaLogit?: number;
  sourceType: string;
  metadata: Record<string, any>;
}

interface GraphEdge {
  id: string;
  from: string;
  to: string;
  relation: string;
}

const CATEGORY_CONFIG: Record<NodeCategory, { icon: React.ReactNode; color: string; bgColor: string; borderColor: string }> = {
  HYPOTHESIS: {
    icon: <Brain size={16} />,
    color: colors.purpleText,
    bgColor: colors.purpleSoft,
    borderColor: colors.purpleBorder,
  },
  EXPERIMENT: {
    icon: <FlaskConical size={16} />,
    color: colors.primary,
    bgColor: colors.accentSoft,
    borderColor: colors.primary,
  },
  COMPONENT: {
    icon: <Layers size={16} />,
    color: colors.infoText,
    bgColor: colors.infoSoft,
    borderColor: colors.infoBorder,
  },
  INTERVENTION: {
    icon: <Zap size={16} />,
    color: colors.warningText,
    bgColor: colors.warningSoft,
    borderColor: colors.warningBorder,
  },
  OBSERVATION: {
    icon: <Eye size={16} />,
    color: colors.bodyMuted,
    bgColor: colors.surfacePearl,
    borderColor: colors.border,
  },
  EVIDENCE: {
    icon: <ShieldCheck size={16} />,
    color: colors.successText,
    bgColor: colors.successSoft,
    borderColor: colors.successBorder,
  },
  FINDING: {
    icon: <Award size={16} />,
    color: colors.dangerText,
    bgColor: colors.dangerSoft,
    borderColor: colors.dangerBorder,
  },
};

const PIPELINE_STAGES: NodeCategory[] = [
  'HYPOTHESIS',
  'EXPERIMENT',
  'COMPONENT',
  'INTERVENTION',
  'OBSERVATION',
  'EVIDENCE',
  'FINDING',
];

export const ResearchGraph: React.FC = () => {
  const {
    activeInvestigation,
    hypotheses,
    runs,
    evidence,
    mechanisms,
    loading,
    error,
    loadEvidence,
    loadRuns,
    loadHypotheses,
  } = useResearchStore();

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [filterCategory, setFilterCategory] = useState<NodeCategory | 'ALL'>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedStages, setExpandedStages] = useState<Set<NodeCategory>>(new Set(PIPELINE_STAGES));

  const graphNodes: GraphNode[] = useMemo(() => {
    const nodes: GraphNode[] = [];

    // Hypothesis nodes
    hypotheses.forEach((h) => {
      nodes.push({
        id: h.id,
        category: 'HYPOTHESIS',
        label: h.title,
        subtitle: h.target_component,
        status: h.status,
        sourceType: 'USER_DEFINED',
        metadata: {
          statement: h.statement,
          prediction: h.prediction,
          falsification_condition: h.falsification_condition,
          evidence_count_supporting: h.evidence_count_supporting,
          evidence_count_contradicting: h.evidence_count_contradicting,
        },
      });
    });

    // Component nodes (from hypothesis targets)
    const componentSet = new Set<string>();
    hypotheses.forEach((h) => {
      if (h.target_component && !componentSet.has(h.target_component)) {
        componentSet.add(h.target_component);
        nodes.push({
          id: `comp_${h.target_component}`,
          category: 'COMPONENT',
          label: h.target_component,
          subtitle: 'Model Component',
          sourceType: 'COMPUTED',
          metadata: { type: 'component' },
        });
      }
    });

    // Experiment/Run nodes
    runs.forEach((r) => {
      nodes.push({
        id: r.id,
        category: 'EXPERIMENT',
        label: `Run ${r.id.slice(-6)}`,
        subtitle: `ΔL = ${r.delta_logit?.toFixed(3) ?? 'N/A'}`,
        status: r.execution_status,
        evidenceLevel: r.used_mock_data ? 'MOCK' : 'LIVE',
        deltaLogit: r.delta_logit,
        sourceType: r.used_mock_data ? 'MOCK_DATA' : 'COMPUTED',
        metadata: {
          model_id: r.model_id,
          execution_time_ms: r.execution_time_ms,
          baseline_logit: r.baseline_logit,
          intervened_logit: r.intervened_logit,
          baseline_target_prob: r.baseline_target_prob,
          intervened_target_prob: r.intervened_target_prob,
          provenance_hash: r.provenance_hash,
        },
      });
    });

    // Evidence nodes
    evidence.forEach((e) => {
      nodes.push({
        id: e.id,
        category: 'EVIDENCE',
        label: e.claim.slice(0, 60),
        subtitle: `${e.metric_name}: ${e.metric_value.toFixed(4)}`,
        evidenceLevel: e.evidence_level,
        supportsHypothesis: e.supports_hypothesis,
        sourceType: e.source_type || 'COMPUTED',
        metadata: {
          metric_name: e.metric_name,
          metric_value: e.metric_value,
          baseline_value: e.baseline_value,
          control_value: e.control_value,
          sample_size: e.sample_size,
          hypothesis_id: e.hypothesis_id,
          experiment_run_id: e.experiment_run_id,
        },
      });
    });

    // Finding nodes (hypotheses with sufficient evidence)
    hypotheses.forEach((h) => {
      if (h.status === 'SUPPORTED') {
        nodes.push({
          id: `finding_${h.id}`,
          category: 'FINDING',
          label: `Finding: ${h.title}`,
          subtitle: h.status,
          evidenceLevel: h.status,
          sourceType: 'INFERENCE',
          metadata: {
            hypothesis_id: h.id,
            statement: h.statement,
            evidence_count_supporting: h.evidence_count_supporting,
          },
        });
      }
    });

    return nodes;
  }, [hypotheses, runs, evidence]);

  const graphEdges: GraphEdge[] = useMemo(() => {
    const edges: GraphEdge[] = [];

    // Hypothesis → Component links
    hypotheses.forEach((h) => {
      if (h.target_component) {
        edges.push({
          id: `e_hyp_comp_${h.id}`,
          from: h.id,
          to: `comp_${h.target_component}`,
          relation: 'targets',
        });
      }
    });

    // Run → Hypothesis links (if run is associated with investigation)
    runs.forEach((r) => {
      hypotheses.forEach((h) => {
        if (r.experiment_id || r.investigation_id) {
          edges.push({
            id: `e_run_hyp_${r.id}_${h.id}`,
            from: r.id,
            to: h.id,
            relation: 'tests',
          });
        }
      });
    });

    // Evidence → Hypothesis links
    evidence.forEach((e) => {
      if (e.hypothesis_id) {
        edges.push({
          id: `e_evid_hyp_${e.id}`,
          from: e.id,
          to: e.hypothesis_id,
          relation: e.supports_hypothesis ? 'supports' : 'contradicts',
        });
      }
      if (e.experiment_run_id) {
        edges.push({
          id: `e_evid_run_${e.id}`,
          from: e.experiment_run_id,
          to: e.id,
          relation: 'produces',
        });
      }
    });

    // Finding → Hypothesis links
    hypotheses.forEach((h) => {
      if (h.status === 'SUPPORTED') {
        edges.push({
          id: `e_find_hyp_${h.id}`,
          from: `finding_${h.id}`,
          to: h.id,
          relation: 'concludes',
        });
      }
    });

    return edges;
  }, [hypotheses, runs, evidence]);

  const filteredNodes = useMemo(() => {
    let nodes = graphNodes;

    if (filterCategory !== 'ALL') {
      nodes = nodes.filter((n) => n.category === filterCategory);
    }

    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      nodes = nodes.filter(
        (n) =>
          n.label.toLowerCase().includes(q) ||
          n.subtitle?.toLowerCase().includes(q) ||
          n.status?.toLowerCase().includes(q)
      );
    }

    return nodes;
  }, [graphNodes, filterCategory, searchQuery]);

  const nodesByCategory = useMemo(() => {
    const groups: Record<NodeCategory, GraphNode[]> = {
      HYPOTHESIS: [],
      EXPERIMENT: [],
      COMPONENT: [],
      INTERVENTION: [],
      OBSERVATION: [],
      EVIDENCE: [],
      FINDING: [],
    };

    filteredNodes.forEach((n) => {
      groups[n.category].push(n);
    });

    return groups;
  }, [filteredNodes]);

  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    return graphNodes.find((n) => n.id === selectedNodeId) || null;
  }, [selectedNodeId, graphNodes]);

  const connectedEdges = useMemo(() => {
    if (!selectedNodeId) return [];
    return graphEdges.filter((e) => e.from === selectedNodeId || e.to === selectedNodeId);
  }, [selectedNodeId, graphEdges]);

  const toggleStage = (stage: NodeCategory) => {
    setExpandedStages((prev) => {
      const next = new Set(prev);
      if (next.has(stage)) {
        next.delete(stage);
      } else {
        next.add(stage);
      }
      return next;
    });
  };

  const getRelationColor = (relation: string) => {
    switch (relation) {
      case 'supports':
        return colors.successText;
      case 'contradicts':
        return colors.dangerText;
      case 'targets':
        return colors.purpleText;
      case 'tests':
        return colors.primary;
      case 'produces':
        return colors.infoText;
      case 'concludes':
        return colors.warningText;
      default:
        return colors.bodyMuted;
    }
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            Research Graph
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Hypothesis → Evidence → Finding Pipeline
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Visualize relationships between hypotheses, experiments, components, and evidence.
          </div>
        </div>

        <button
          onClick={() => {
            if (activeInvestigation?.id) {
              loadEvidence(activeInvestigation.id);
              loadRuns(activeInvestigation.id);
              loadHypotheses(activeInvestigation.id);
            }
          }}
          disabled={loading}
          style={{
            padding: '8px 12px',
            borderRadius: 6,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.canvas,
            color: colors.bodyMuted,
            fontSize: 12,
            fontWeight: 600,
            cursor: loading ? 'default' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} /> Refresh
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div style={{
          padding: '10px 14px',
          borderRadius: 8,
          backgroundColor: colors.dangerSoft,
          border: `1px solid ${colors.dangerBorder}`,
          color: colors.dangerText,
          fontSize: 12,
          marginBottom: 16,
        }}>
          {error}
        </div>
      )}

      {/* Search and Filter */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: colors.bodyMuted }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search nodes..."
            style={{
              width: '100%',
              padding: '8px 12px 8px 32px',
              borderRadius: 6,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.ink,
              fontSize: 12,
              boxSizing: 'border-box',
              outline: 'none',
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: 4 }}>
          <button
            onClick={() => setFilterCategory('ALL')}
            style={{
              padding: '6px 12px',
              borderRadius: 6,
              border: `1px solid ${filterCategory === 'ALL' ? colors.primary : colors.border}`,
              backgroundColor: filterCategory === 'ALL' ? colors.accentSoft : colors.canvas,
              color: filterCategory === 'ALL' ? colors.primary : colors.bodyMuted,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            ALL
          </button>
          {PIPELINE_STAGES.map((cat) => (
            <button
              key={cat}
              onClick={() => setFilterCategory(cat)}
              style={{
                padding: '6px 12px',
                borderRadius: 6,
                border: `1px solid ${filterCategory === cat ? CATEGORY_CONFIG[cat].borderColor : colors.border}`,
                backgroundColor: filterCategory === cat ? CATEGORY_CONFIG[cat].bgColor : colors.canvas,
                color: filterCategory === cat ? CATEGORY_CONFIG[cat].color : colors.bodyMuted,
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedNode ? '2fr 1fr' : '1fr', gap: 16 }}>
        {/* Pipeline View */}
        <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 16, textTransform: 'uppercase', letterSpacing: 0.4 }}>
            Research Pipeline ({filteredNodes.length} nodes)
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {PIPELINE_STAGES.map((stage, idx) => {
              const stageNodes = nodesByCategory[stage];
              const isExpanded = expandedStages.has(stage);
              const config = CATEGORY_CONFIG[stage];

              if (filterCategory !== 'ALL' && filterCategory !== stage) {
                return null;
              }

              return (
                <React.Fragment key={stage}>
                  {/* Stage Header */}
                  <div
                    onClick={() => toggleStage(stage)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '10px 12px',
                      borderRadius: 8,
                      backgroundColor: config.bgColor,
                      border: `1px solid ${config.borderColor}`,
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ color: config.color }}>{config.icon}</div>
                      <span style={{ fontSize: 13, fontWeight: 700, color: config.color }}>{stage}</span>
                      <span style={{
                        fontSize: 11,
                        padding: '2px 8px',
                        borderRadius: 10,
                        backgroundColor: 'rgba(0,0,0,0.1)',
                        color: config.color,
                        fontWeight: 700,
                      }}>
                        {stageNodes.length}
                      </span>
                    </div>
                    {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </div>

                  {/* Stage Nodes */}
                  {isExpanded && stageNodes.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: '8px 0 8px 24px' }}>
                      {stageNodes.map((node) => (
                        <div
                          key={node.id}
                          onClick={() => setSelectedNodeId(node.id)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '10px 12px',
                            borderRadius: 8,
                            border: `1px solid ${selectedNodeId === node.id ? config.borderColor : colors.border}`,
                            backgroundColor: selectedNodeId === node.id ? config.bgColor : colors.canvas,
                            cursor: 'pointer',
                            boxShadow: selectedNodeId === node.id ? `0 0 0 2px ${config.borderColor}` : 'none',
                            transition: 'all 0.15s ease',
                          }}
                        >
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: 12, fontWeight: 600, color: colors.ink }}>{node.label}</div>
                            {node.subtitle && (
                              <div style={{ fontSize: 11, color: colors.bodyMuted, marginTop: 2 }}>{node.subtitle}</div>
                            )}
                          </div>

                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            {node.status && (
                              <span style={{
                                fontSize: 9,
                                fontWeight: 700,
                                padding: '2px 6px',
                                borderRadius: 4,
                                backgroundColor: config.bgColor,
                                color: config.color,
                              }}>
                                {node.status}
                              </span>
                            )}
                            {node.evidenceLevel && (
                              <span style={{
                                fontSize: 9,
                                fontWeight: 700,
                                padding: '2px 6px',
                                borderRadius: 4,
                                backgroundColor: colors.surfacePearl,
                                color: colors.bodyMuted,
                              }}>
                                {node.evidenceLevel}
                              </span>
                            )}
                            {node.supportsHypothesis !== undefined && (
                              <span style={{
                                fontSize: 9,
                                fontWeight: 700,
                                padding: '2px 6px',
                                borderRadius: 4,
                                backgroundColor: node.supportsHypothesis ? colors.successSoft : colors.dangerSoft,
                                color: node.supportsHypothesis ? colors.successText : colors.dangerText,
                              }}>
                                {node.supportsHypothesis ? 'SUPPORTS' : 'CONTRADICTS'}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Stage Empty */}
                  {isExpanded && stageNodes.length === 0 && (
                    <div style={{ padding: '12px 24px', fontSize: 12, color: colors.bodyMuted }}>
                      No {stage.toLowerCase()} nodes found.
                    </div>
                  )}

                  {/* Pipeline Arrow */}
                  {idx < PIPELINE_STAGES.length - 1 && filterCategory === 'ALL' && (
                    <div style={{ display: 'flex', justifyContent: 'center', padding: '4px 0' }}>
                      <ArrowDown size={16} style={{ color: colors.border }} />
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Node Inspector */}
        {selectedNode && (
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: colors.ink, textTransform: 'uppercase' }}>
                Node Inspector
              </span>
              <button
                onClick={() => setSelectedNodeId(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 16, color: colors.bodyMuted }}
              >
                ×
              </button>
            </div>

            {/* Node Header */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: 10,
              borderRadius: 8,
              backgroundColor: CATEGORY_CONFIG[selectedNode.category].bgColor,
              border: `1px solid ${CATEGORY_CONFIG[selectedNode.category].borderColor}`,
              marginBottom: 12,
            }}>
              <div style={{ color: CATEGORY_CONFIG[selectedNode.category].color }}>
                {CATEGORY_CONFIG[selectedNode.category].icon}
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: CATEGORY_CONFIG[selectedNode.category].color }}>
                  {selectedNode.label}
                </div>
                {selectedNode.subtitle && (
                  <div style={{ fontSize: 11, color: colors.bodyMuted }}>{selectedNode.subtitle}</div>
                )}
              </div>
            </div>

            {/* Metadata */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                <b>ID:</b> <code>{selectedNode.id}</code>
              </div>
              <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                <b>Category:</b> {selectedNode.category}
              </div>
              <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                <b>Source:</b> {selectedNode.sourceType}
              </div>
              {selectedNode.status && (
                <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                  <b>Status:</b> {selectedNode.status}
                </div>
              )}

              {/* Detailed Metadata */}
              <div style={{ marginTop: 8, padding: 10, borderRadius: 6, backgroundColor: colors.surfacePearl, border: `1px solid ${colors.border}` }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, marginBottom: 6 }}>Details</div>
                {Object.entries(selectedNode.metadata).map(([key, value]) => {
                  if (value === undefined || value === null) return null;
                  return (
                    <div key={key} style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 4 }}>
                      <b>{key.replace(/_/g, ' ')}:</b>{' '}
                      {typeof value === 'number' ? value.toFixed(4) : String(value)}
                    </div>
                  );
                })}
              </div>

              {/* Connected Edges */}
              {connectedEdges.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, marginBottom: 6 }}>Connections</div>
                  {connectedEdges.map((edge) => (
                    <div
                      key={edge.id}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        padding: '4px 8px',
                        marginBottom: 4,
                        borderRadius: 4,
                        backgroundColor: colors.canvas,
                        border: `1px solid ${colors.border}`,
                        fontSize: 11,
                      }}
                    >
                      <span style={{ color: colors.bodyMuted }}>{edge.from.slice(0, 8)}...</span>
                      <span style={{ fontWeight: 700, color: getRelationColor(edge.relation) }}>
                        → {edge.relation} →
                      </span>
                      <span style={{ color: colors.bodyMuted }}>{edge.to.slice(0, 8)}...</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
