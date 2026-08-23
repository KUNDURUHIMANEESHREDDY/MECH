import React, { useEffect, useMemo, useState } from 'react';
import { 
  Share2, 
  Play, 
  Loader2, 
  AlertTriangle, 
  ShieldCheck, 
  ShieldAlert, 
  ArrowRight, 
  ArrowLeft, 
  Layers, 
  Eye, 
  Zap,
  Target,
  Info,
  Sliders,
  CheckCircle2,
  AlertCircle,
  GitCommit,
  GitMerge,
  Cpu,
  RefreshCw,
  BarChart
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useModel } from '../../shared/hooks/useModel';
import { scienceApi } from '../../science/api/scienceApi';
import { 
  UnifiedScientificReport, 
  CircuitPathwayEdge, 
  EvidenceLevel, 
  MechanismType,
  CircuitCompositionReport,
  PathwayVerificationReport
} from '../../science/types/scientificTypes';
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
const selectStyle: React.CSSProperties = {
  padding: '5px 8px',
  borderRadius: 6,
  border: `1px solid ${colors.hairline}`,
  background: colors.canvas,
  fontSize: 12,
  color: colors.ink,
};

export const KnowledgeGraphPanel: FC<PanelContext> = () => {
  const { state: model, listModels } = useModel();
  const [prompt, setPrompt] = useState('The capital of France is');
  const [targetToken, setTargetToken] = useState(' Paris');
  const [report, setReport] = useState<UnifiedScientificReport | null>(null);
  const [pathwayReport, setPathwayReport] = useState<PathwayVerificationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [verifyingPath, setVerifyingPath] = useState(false);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [edgeOrientation, setEdgeOrientation] = useState<'value_flow' | 'query_relation'>('value_flow');
  const [selectedLensLayer, setSelectedLensLayer] = useState<number | null>(null);

  useEffect(() => {
    if (model.availableModels.length === 0 && !model.loading) void listModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRunInvestigation = async () => {
    setLoading(true);
    try {
      const rep = await scienceApi.fetchUnifiedInvestigationReport({
        clean_prompt: prompt,
        corrupted_prompt: 'The capital of Germany is',
        target_token: targetToken,
        model_id: model.modelInfo?.model_name || 'gpt2',
      });
      setReport(rep);
      setSelectedLensLayer(rep.predictive_divergence_layer);
      if (rep.circuit_edges.length > 0) {
        setSelectedEdgeId(rep.circuit_edges[0].edge_id);
      }

      // Automatically run End-to-End Pathway Verification & Mediation Rescue
      void runPathwayVerification(rep.predictive_divergence_layer || 8);
    } catch (e) {
      console.error('Failed to load scientific knowledge graph:', e);
    } finally {
      setLoading(false);
    }
  };

  const runPathwayVerification = async (layer: number = 8) => {
    setVerifyingPath(true);
    try {
      const pRep = await scienceApi.verifyFullPathway({
        clean_prompt: prompt,
        target_token: targetToken,
        layer: layer,
        model_id: model.modelInfo?.model_name || 'gpt2',
      });
      setPathwayReport(pRep);
    } catch (err) {
      console.warn('Full pathway verification warning:', err);
    } finally {
      setVerifyingPath(false);
    }
  };

  useEffect(() => {
    void handleRunInvestigation();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedEdge = useMemo(() => {
    if (!report || !selectedEdgeId) return null;
    return report.circuit_edges.find((e) => e.edge_id === selectedEdgeId) || report.circuit_edges[0];
  }, [report, selectedEdgeId]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>
      {/* Top Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Share2 size={18} color={colors.primary} />
          <div>
            <div style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Evidence-Aware Mechanistic Knowledge Graph</div>
            <div style={{ fontSize: 11, color: colors.bodyMuted }}>
              Edge vs Pathway Causality • Mediation Rescue ($A \to B \to Y$) • Empirical Null Model
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            style={{ ...selectStyle, width: 220 }}
            placeholder="Clean prompt..."
          />
          <button onClick={handleRunInvestigation} disabled={loading} style={{ ...btn, opacity: loading ? 0.5 : 1 }}>
            {loading ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
            {loading ? 'Analyzing…' : 'Investigate'}
          </button>
        </div>
      </div>

      {/* Dual Badge Level Banner: Edge-Level vs End-to-End Pathway Level */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 10 }}>
        {/* Edge-Level Causality Banner */}
        <div style={{ ...card, borderLeft: '4px solid #3b82f6', background: 'rgba(59, 130, 246, 0.04)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <GitCommit size={15} color="#3b82f6" />
              <strong style={{ color: colors.ink, fontSize: 12 }}>Edge-Level Causality</strong>
            </div>
            <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 6px', background: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6', borderRadius: 4 }}>
              [INDIVIDUAL EDGES TESTED]
            </span>
          </div>
          <div style={{ fontSize: 11, color: colors.bodyMuted }}>
            Evaluates single-hop connection strength: "Does this individual component / attention routing edge transmit information?"
          </div>
        </div>

        {/* End-to-End Pathway Level Causality Banner */}
        <div style={{ ...card, borderLeft: `4px solid ${pathwayReport?.path_causal_status === 'END_TO_END_VERIFIED' ? '#10b981' : '#f59e0b'}`, background: colors.purpleSoft }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <GitMerge size={15} color={colors.primary} />
              <strong style={{ color: colors.ink, fontSize: 12 }}>End-to-End Pathway Causality</strong>
            </div>
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                padding: '2px 6px',
                background: pathwayReport?.path_causal_status === 'END_TO_END_VERIFIED' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                color: pathwayReport?.path_causal_status === 'END_TO_END_VERIFIED' ? '#10b981' : '#f59e0b',
                borderRadius: 4,
              }}
            >
              [{pathwayReport?.path_causal_status?.replace('_', ' ') || 'VERIFYING'}]
            </span>
          </div>
          <div style={{ fontSize: 11, color: colors.bodyMuted }}>
            Multi-hop composite intervention: "Does intervening on the complete transmission chain mediate downstream behavior?"
          </div>
        </div>
      </div>

      {/* Mediation Rescue Experiment Card */}
      {pathwayReport?.mediation_rescue && (
        <div style={{ ...card, border: '1px solid #10b981', background: 'rgba(16, 185, 129, 0.03)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: `1px solid ${colors.hairline}`, paddingBottom: 6 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <RefreshCw size={15} color="#10b981" />
              <div>
                <span style={{ fontWeight: 700, color: colors.ink, fontSize: 13 }}>
                  Formal Mediation Rescue Experiment: {pathwayReport.mediation_rescue.source_node} ➔ {pathwayReport.mediation_rescue.mediator_node} ➔ Target
                </span>
              </div>
            </div>
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                padding: '2px 8px',
                borderRadius: 4,
                background: pathwayReport.mediation_rescue.formal_mediation_status === 'CONFIRMED_CAUSAL_MEDIATOR' ? '#10b981' : '#f59e0b',
                color: '#fff',
              }}
            >
              {pathwayReport.mediation_rescue.formal_mediation_status.replace(/_/g, ' ')}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8, fontSize: 11 }}>
            <div style={{ padding: '6px 8px', background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 6 }}>
              <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Source A Knockout Logit</div>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#ef4444' }}>
                {pathwayReport.mediation_rescue.ablated_source_logit.toFixed(2)}
              </div>
            </div>
            <div style={{ padding: '6px 8px', background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 6 }}>
              <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Mediator Clamped Logit</div>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#10b981' }}>
                {pathwayReport.mediation_rescue.rescued_logit.toFixed(2)}
              </div>
            </div>
            <div style={{ padding: '6px 8px', background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 6 }}>
              <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Mediation Rescue Recovery</div>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#10b981' }}>
                +{(pathwayReport.mediation_rescue.rescue_delta_recovery).toFixed(2)} Δz
              </div>
            </div>
            <div style={{ padding: '6px 8px', background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 6 }}>
              <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Rescue Fraction</div>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#10b981' }}>
                {(pathwayReport.mediation_rescue.rescue_fraction * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div style={{ fontSize: 11, color: colors.bodyMuted }}>
            <strong>Causal Mediation Diagnostic: </strong>{pathwayReport.mediation_rescue.rescue_verdict}
          </div>
        </div>
      )}

      {/* Path-Level Empirical Null-Distribution Statistics Box */}
      {pathwayReport?.null_distribution && (
        <div style={card}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', alignItems: 'center', gap: 6 }}>
              <BarChart size={15} color={colors.primary} />
              <span>Path-Level Empirical Null Distribution (K={pathwayReport.null_distribution.control_path_count} Matched Paths)</span>
            </div>
            <span style={{ fontSize: 11, fontWeight: 700, color: '#10b981' }}>
              Rank: {pathwayReport.null_distribution.observed_path_percentile.toFixed(0)}th Percentile of Null Model
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8, fontSize: 11 }}>
            <div style={{ padding: '6px 8px', background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 6 }}>
              <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Observed Path Δz</div>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#10b981' }}>+{pathwayReport.composite_path_effect.toFixed(2)}</div>
            </div>
            <div style={{ padding: '6px 8px', background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 6 }}>
              <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Null Mean Δz</div>
              <div style={{ fontWeight: 700, fontSize: 13, color: colors.bodyMuted }}>+{pathwayReport.null_distribution.mean_null_delta.toFixed(2)}</div>
            </div>
            <div style={{ padding: '6px 8px', background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 6 }}>
              <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Null Max Δz</div>
              <div style={{ fontWeight: 700, fontSize: 13, color: colors.bodyMuted }}>+{pathwayReport.null_distribution.max_null_delta.toFixed(2)}</div>
            </div>
            <div style={{ padding: '6px 8px', background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 6 }}>
              <div style={{ color: colors.bodyMuted, fontSize: 10 }}>Empirical p-value</div>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#10b981' }}>p = {pathwayReport.null_distribution.empirical_p_value.toFixed(3)}</div>
            </div>
          </div>

          {/* Null Control Path Distribution Trace */}
          <div style={{ display: 'flex', gap: 4, alignItems: 'center', overflowX: 'auto', fontSize: 10.5, color: colors.bodyMuted }}>
            <span>Control Path Deltas:</span>
            {pathwayReport.null_distribution.control_path_deltas.map((d, dIdx) => (
              <span key={dIdx} style={{ padding: '2px 5px', background: 'rgba(0,0,0,0.04)', borderRadius: 3, fontFamily: 'monospace' }}>
                +{d.toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* End-to-End Circuit Verification Studio */}
      {pathwayReport && (
        <div style={{ ...card, border: `1px solid ${colors.primary}`, background: colors.canvas }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: `1px solid ${colors.hairline}`, paddingBottom: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Cpu size={16} color={colors.primary} />
              <div>
                <span style={{ fontWeight: 700, color: colors.ink, fontSize: 13 }}>
                  Multi-Hop Pathway Verification Battery
                </span>
                <span style={{ fontSize: 11, color: colors.bodyMuted, marginLeft: 8 }}>
                  Chain: {pathwayReport.node_chain.join(' ➔ ')}
                </span>
              </div>
            </div>

            <button
              onClick={() => runPathwayVerification(report?.predictive_divergence_layer || 8)}
              disabled={verifyingPath}
              style={{ ...btn, padding: '4px 10px', fontSize: 11 }}
            >
              {verifyingPath ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
              {verifyingPath ? 'Patching…' : 'Re-Run Path Battery'}
            </button>
          </div>

          {/* 6-Step Experimental Comparison Table */}
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, marginTop: 4 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${colors.hairline}`, textAlign: 'left', color: colors.bodyMuted }}>
                <th style={{ padding: '5px 6px' }}>Experimental Step</th>
                <th style={{ padding: '5px 6px' }}>Intervention Target</th>
                <th style={{ padding: '5px 6px' }}>Type</th>
                <th style={{ padding: '5px 6px', textAlign: 'right' }}>Δ Logit</th>
                <th style={{ padding: '5px 6px', textAlign: 'right' }}>Δ Prob</th>
                <th style={{ padding: '5px 6px' }}>Epistemic Diagnostic</th>
              </tr>
            </thead>
            <tbody>
              {pathwayReport.step_measurements.map((step, sIdx) => (
                <tr key={sIdx} style={{ borderBottom: `1px solid ${colors.hairline}` }}>
                  <td style={{ padding: '5px 6px', fontWeight: 600, color: colors.ink }}>{step.step_name}</td>
                  <td style={{ padding: '5px 6px', fontFamily: 'monospace', color: colors.primary }}>{step.intervention_target}</td>
                  <td style={{ padding: '5px 6px', color: colors.bodyMuted }}>{step.intervention_type}</td>
                  <td style={{ padding: '5px 6px', textAlign: 'right', fontWeight: 700, color: step.delta_logit > 0.1 ? '#10b981' : colors.ink }}>
                    +{step.delta_logit.toFixed(2)}
                  </td>
                  <td style={{ padding: '5px 6px', textAlign: 'right', color: colors.bodyMuted }}>
                    {(step.delta_prob * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '5px 6px', color: colors.bodyMuted }}>{step.description}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pathway Verdict */}
          <div style={{ padding: '6px 10px', background: 'rgba(16, 185, 129, 0.08)', borderRadius: 4, fontSize: 11, color: '#065f46' }}>
            <strong>Path Verification Verdict: </strong>{pathwayReport.path_verdict}
          </div>
        </div>
      )}

      {/* Logit Lens Depth Microscope Track */}
      {report?.logit_lens_trajectory && (
        <div style={card}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontWeight: 700, color: colors.ink, display: 'flex', alignItems: 'center', gap: 6 }}>
              <Eye size={14} color={colors.primary} />
              <span>Logit Lens Temporal Depth Microscope</span>
            </div>
            <span style={{ fontSize: 11, color: colors.bodyMuted }}>
              Emergence Point: Layer {report.predictive_divergence_layer} (Investigation Seed)
            </span>
          </div>

          <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
            {report.logit_lens_trajectory.map((step) => {
              const isSelected = selectedLensLayer === step.layer;
              const isTransition = step.is_predictive_transition;
              return (
                <div
                  key={step.layer}
                  onClick={() => setSelectedLensLayer(step.layer)}
                  style={{
                    flex: '0 0 auto',
                    padding: '6px 8px',
                    borderRadius: 6,
                    border: isSelected ? `1px solid ${colors.primary}` : isTransition ? '1px solid #f59e0b' : `1px solid ${colors.hairline}`,
                    background: isSelected ? colors.purpleSoft : isTransition ? 'rgba(245, 158, 11, 0.08)' : colors.canvas,
                    cursor: 'pointer',
                    textAlign: 'center',
                    minWidth: 70,
                  }}
                >
                  <div style={{ fontSize: 9.5, color: colors.bodyMuted }}>L{step.layer}</div>
                  <div style={{ fontWeight: 700, fontSize: 11, color: isTransition ? '#f59e0b' : colors.ink }}>
                    '{step.top_token}'
                  </div>
                  <div style={{ fontSize: 9.5, color: '#10b981' }}>{(step.probability * 100).toFixed(0)}%</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Two-Way Edge Orientation Controls & Mechanism Filters */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
        <div style={{ display: 'flex', gap: 6, background: 'rgba(0,0,0,0.05)', padding: 2, borderRadius: 6 }}>
          <button
            onClick={() => setEdgeOrientation('value_flow')}
            style={{
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 600,
              border: 'none',
              borderRadius: 4,
              background: edgeOrientation === 'value_flow' ? colors.primary : 'transparent',
              color: edgeOrientation === 'value_flow' ? '#fff' : colors.bodyMuted,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <ArrowRight size={12} /> Forward Value Flow (A ➔ B)
          </button>
          <button
            onClick={() => setEdgeOrientation('query_relation')}
            style={{
              padding: '4px 10px',
              fontSize: 11,
              fontWeight: 600,
              border: 'none',
              borderRadius: 4,
              background: edgeOrientation === 'query_relation' ? '#10b981' : 'transparent',
              color: edgeOrientation === 'query_relation' ? '#fff' : colors.bodyMuted,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <ArrowLeft size={12} /> Backward Attention Querying (B ➔ A)
          </button>
        </div>

        <div style={{ display: 'flex', gap: 6, fontSize: 11 }}>
          <span style={{ color: colors.primary }}>● Attention Routing (OV/QK)</span>
          <span style={{ color: '#f59e0b' }}>● MLP Associative Projection</span>
          <span style={{ color: '#10b981' }}>● Residual Stream Accumulation</span>
        </div>
      </div>

      {/* Main Graph & Full Structured Edge Evidence Inspector */}
      {report && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 12 }}>
          {/* Circuit Graph Directed Edge List */}
          <div style={card}>
            <div style={{ fontWeight: 700, color: colors.ink }}>
              Multi-Mechanism Pathway Edges ({report.circuit_edges.length})
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {report.circuit_edges.map((edge) => {
                const isSelected = selectedEdge?.edge_id === edge.edge_id;
                let mechColor: string = colors.primary;
                if (edge.mechanism_type === 'mlp_projection') mechColor = '#f59e0b';
                if (edge.mechanism_type === 'residual_stream') mechColor = '#10b981';

                const isWeakest = report.circuit_composition?.weakest_link_edge_id === edge.edge_id;

                return (
                  <div
                    key={edge.edge_id}
                    onClick={() => setSelectedEdgeId(edge.edge_id)}
                    style={{
                      padding: '8px 10px',
                      borderRadius: 6,
                      border: isSelected ? `2px solid ${mechColor}` : isWeakest ? '1px dashed #f59e0b' : `1px solid ${colors.hairline}`,
                      background: isSelected ? 'rgba(124, 58, 237, 0.08)' : colors.canvas,
                      cursor: 'pointer',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 600, color: colors.ink, fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span>{edge.source_id} ➔ {edge.target_id}</span>
                        {isWeakest && (
                          <span style={{ fontSize: 9.5, fontWeight: 700, color: '#f59e0b', background: 'rgba(245, 158, 11, 0.15)', padding: '1px 4px', borderRadius: 3 }}>
                            WEAKEST LINK
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: 10.5, color: colors.bodyMuted, marginTop: 2 }}>
                        {edgeOrientation === 'value_flow' ? edge.value_flow_description : edge.query_relation_description}
                      </div>
                    </div>

                    <div style={{ textAlign: 'right' }}>
                      <span
                        style={{
                          fontSize: 9.5,
                          fontWeight: 700,
                          padding: '2px 6px',
                          borderRadius: 4,
                          background: edge.evidence_level === 'CAUSALLY_VERIFIED'
                            ? 'rgba(16, 185, 129, 0.15)'
                            : edge.evidence_level === 'SUPPORTED'
                            ? 'rgba(59, 130, 246, 0.15)'
                            : 'rgba(245, 158, 11, 0.15)',
                          color: edge.evidence_level === 'CAUSALLY_VERIFIED'
                            ? '#10b981'
                            : edge.evidence_level === 'SUPPORTED'
                            ? '#3b82f6'
                            : '#f59e0b',
                        }}
                      >
                        [{edge.evidence_level}]
                      </span>
                      <div style={{ fontSize: 10, color: colors.bodyMuted, marginTop: 2 }}>
                        Routing: {edge.attention_routing_score !== undefined ? (edge.attention_routing_score * 100).toFixed(0) + '%' : 'N/A'}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Full Structured Edge Evidence Object Deep-Dive Inspector */}
          {selectedEdge && (
            <div style={card}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 700, color: colors.ink, fontSize: 13 }}>
                  Edge Evidence Object: {selectedEdge.source_id} ➔ {selectedEdge.target_id}
                </span>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 4,
                    background: selectedEdge.evidence_level === 'CAUSALLY_VERIFIED' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(59, 130, 246, 0.2)',
                    color: selectedEdge.evidence_level === 'CAUSALLY_VERIFIED' ? '#10b981' : '#3b82f6',
                  }}
                >
                  [{selectedEdge.evidence_level}]
                </span>
              </div>

              <div style={{ fontSize: 11.5, color: colors.bodyMuted }}>
                <strong>Mechanism: </strong>{selectedEdge.mechanism_type.replace('_', ' ').toUpperCase()}
              </div>

              {/* Orientation Explanations */}
              <div style={{ padding: 8, background: 'rgba(0,0,0,0.03)', borderRadius: 6, border: `1px solid ${colors.hairline}` }}>
                <div style={{ fontSize: 11, marginBottom: 4 }}>
                  <strong style={{ color: colors.primary }}>Forward Value Flow:</strong>
                  <div style={{ color: colors.ink, marginTop: 2 }}>{selectedEdge.value_flow_description}</div>
                </div>
                <div style={{ fontSize: 11, marginTop: 6, paddingTop: 6, borderTop: `1px solid ${colors.hairline}` }}>
                  <strong style={{ color: '#10b981' }}>Backward Querying Mechanism:</strong>
                  <div style={{ color: colors.ink, marginTop: 2 }}>{selectedEdge.query_relation_description}</div>
                </div>
              </div>

              {/* Edge Evidence Object Data Breakdown */}
              {selectedEdge.evidence_object && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11 }}>
                  <div style={{ fontWeight: 600, color: colors.ink }}>Structured Edge Evidence:</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                    <div style={{ padding: 6, background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 4 }}>
                      <span style={{ color: colors.bodyMuted }}>Logit Lens Stage:</span>
                      <div style={{ fontWeight: 600, color: colors.ink }}>{selectedEdge.evidence_object.logit_lens_stage}</div>
                    </div>
                    <div style={{ padding: 6, background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 4 }}>
                      <span style={{ color: colors.bodyMuted }}>Causal Mediation Δz:</span>
                      <div style={{ fontWeight: 700, color: '#10b981' }}>
                        {selectedEdge.causal_mediation_effect ? `+${(selectedEdge.causal_mediation_effect).toFixed(2)}` : 'Pending'}
                      </div>
                    </div>
                    <div style={{ padding: 6, background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 4 }}>
                      <span style={{ color: colors.bodyMuted }}>Attention Routing Score:</span>
                      <div style={{ fontWeight: 700, color: colors.ink }}>
                        {selectedEdge.attention_routing_score !== undefined ? (selectedEdge.attention_routing_score * 100).toFixed(1) + '%' : 'N/A'}
                      </div>
                    </div>
                    <div style={{ padding: 6, background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 4 }}>
                      <span style={{ color: colors.bodyMuted }}>Robust Specificity:</span>
                      <div style={{ fontWeight: 700, color: '#10b981' }}>
                        {selectedEdge.evidence_object.robust_specificity ? `${selectedEdge.evidence_object.robust_specificity.toFixed(1)}x` : 'N/A'}
                      </div>
                    </div>
                  </div>

                  <div style={{ padding: 6, background: 'rgba(59, 130, 246, 0.05)', borderRadius: 4, border: '1px solid rgba(59, 130, 246, 0.2)', fontSize: 10.5, color: colors.bodyMuted }}>
                    <strong>Epistemic Scope: </strong>{selectedEdge.evidence_object.epistemic_scope}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Falsification Audits Box */}
      {report?.falsified_hypotheses && report.falsified_hypotheses.length > 0 && (
        <div style={{ ...card, borderLeft: '4px solid #ef4444' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontWeight: 700, color: '#ef4444', display: 'flex', alignItems: 'center', gap: 6 }}>
              <ShieldAlert size={15} />
              <span>Causal Falsification Audits (Attention ≠ Causality Proof)</span>
            </div>
            <span style={{ fontSize: 10, fontWeight: 700, color: '#ef4444', padding: '2px 6px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: 4 }}>
              Falsified Pathways
            </span>
          </div>

          <div style={{ fontSize: 11, color: colors.bodyMuted }}>
            The following connections showed high observational attention weights but failed double-counterfactual path patching:
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {report.falsified_hypotheses.map((f, idx) => (
              <div key={idx} style={{ padding: '8px 10px', background: 'rgba(239, 68, 68, 0.04)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, fontWeight: 600 }}>
                  <span style={{ color: colors.ink }}>Pathway: {f.pathway}</span>
                  <span style={{ color: '#ef4444' }}>Routing: {f.attention_routing_score !== undefined ? (f.attention_routing_score * 100).toFixed(0) + '%' : 'N/A'} | Causal Δ: {f.causal_effect !== undefined ? (f.causal_effect * 100).toFixed(0) + '%' : 'N/A'}</span>
                </div>
                <div style={{ fontSize: 11, color: colors.bodyMuted, marginTop: 3 }}>
                  {f.falsification_verdict}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};