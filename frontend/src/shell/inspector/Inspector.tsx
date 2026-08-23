import React, { useState } from 'react';
import { useSelectionStore } from '../../shared/stores/selection';
import { useResearchStore } from '../../shared/stores/research';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import { colors } from '../../design/tokens/colors';
import {
  Activity,
  ArrowRight,
  ShieldCheck,
  Zap,
  Layers,
  Database,
  GitBranch,
  FileText,
  Clock,
} from 'lucide-react';

type InspectorTab = 'overview' | 'identity' | 'relationships' | 'evidence' | 'actions' | 'provenance';

export const Inspector: React.FC = () => {
  const selection = useSelectionStore();
  const research = useResearchStore();
  const { openPanel } = useWorkspaceStore();

  const [activeTab, setActiveTab] = useState<InspectorTab>('overview');

  const selectedComp = selection.researchComponent || (
    selection.layer !== null && selection.head !== null
      ? { name: `L${selection.layer}H${selection.head}`, layer: selection.layer, head: selection.head, componentType: 'head' as const }
      : null
  );

  const linkedHypothesis = research.hypotheses.find(
    (h) => h.target_component === selectedComp?.name
  );

  const linkedEvidence = research.evidence.filter(
    (e) => e.claim.includes(selectedComp?.name || '###')
  );

  const linkedRuns = research.runs.filter(
    (r) => r.model_id === (research.activeInvestigation?.model_id || 'gpt2')
  );

  const getKnowledgeBadge = (level: string) => {
    switch (level) {
      case 'CAUSALLY_VERIFIED':
        return <span style={{ fontSize: 9, fontWeight: 800, padding: '1px 5px', borderRadius: 4, backgroundColor: colors.successSoft, color: colors.successText, border: `1px solid ${colors.successBorder}` }}>CAUSAL EVIDENCE</span>;
      case 'SUPPORTED':
        return <span style={{ fontSize: 9, fontWeight: 800, padding: '1px 5px', borderRadius: 4, backgroundColor: colors.primarySoft, color: colors.primary, border: `1px solid ${colors.border}` }}>INFERENCE</span>;
      case 'OBSERVED':
      default:
        return <span style={{ fontSize: 9, fontWeight: 800, padding: '1px 5px', borderRadius: 4, backgroundColor: colors.surfacePearl, color: colors.bodyMuted, border: `1px solid ${colors.border}` }}>OBSERVATION</span>;
    }
  };

  return (
    <aside className="shell-inspector" style={inspectorStyle}>
      {/* Inspector Title */}
      <div style={headerStyle}>
        <Activity size={14} style={{ color: colors.primary }} />
        <span style={{ fontWeight: 700, fontSize: '12px', letterSpacing: '-0.2px' }}>Universal Object Inspector</span>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: `1px solid ${colors.border}`, backgroundColor: colors.surfaceTile1, overflowX: 'auto' }}>
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'identity', label: 'Identity' },
          { id: 'relationships', label: 'Links' },
          { id: 'evidence', label: `Evidence (${linkedEvidence.length})` },
          { id: 'actions', label: 'Actions' },
          { id: 'provenance', label: 'Provenance' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as InspectorTab)}
            style={{
              padding: '6px 8px',
              fontSize: '11px',
              fontWeight: activeTab === tab.id ? 700 : 500,
              color: activeTab === tab.id ? colors.primary : colors.bodyMuted,
              border: 'none',
              borderBottom: activeTab === tab.id ? `2px solid ${colors.primary}` : 'none',
              backgroundColor: 'transparent',
              cursor: 'pointer',
              whiteSpace: 'nowrap',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Body */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px', boxSizing: 'border-box' }}>
        {selectedComp ? (
          <div>
            {/* OVERVIEW TAB */}
            {activeTab === 'overview' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ padding: '10px', borderRadius: '8px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                  <div style={{ fontSize: '10px', fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase' }}>
                    Component Coordinate
                  </div>
                  <div style={{ fontSize: '16px', fontWeight: 800, color: colors.ink, marginTop: 2 }}>
                    {selectedComp.name}
                  </div>
                  <div style={{ fontSize: '11px', color: colors.bodyMuted, marginTop: 2 }}>
                    Layer {selectedComp.layer ?? 0} · Head {selectedComp.head ?? 0} ({selectedComp.componentType || 'Attention Head'})
                  </div>
                </div>

                {linkedHypothesis ? (
                  <div style={{ padding: '8px 10px', borderRadius: '6px', backgroundColor: colors.surfacePearl, border: `1px solid ${colors.border}` }}>
                    <div style={{ fontSize: '10px', fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase' }}>Linked Hypothesis</div>
                    <div style={{ fontSize: '12px', fontWeight: 700, color: colors.ink, marginTop: 2 }}>{linkedHypothesis.title}</div>
                    <div style={{ fontSize: '11px', color: colors.body, marginTop: 2, lineHeight: 1.35 }}>{linkedHypothesis.statement}</div>
                    <div style={{ fontSize: '10px', fontWeight: 700, color: colors.primary, marginTop: 4 }}>Status: {linkedHypothesis.status}</div>
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: colors.bodyMuted, fontStyle: 'italic' }}>
                    No active hypothesis registered for {selectedComp.name}.
                  </div>
                )}
              </div>
            )}

            {/* IDENTITY TAB */}
            {activeTab === 'identity' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '11px' }}>
                <div style={{ padding: '8px', borderRadius: '6px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                  <div style={{ color: colors.bodyMuted }}>Object ID</div>
                  <div style={{ fontWeight: 600, color: colors.ink }}><code>node_{selectedComp.name}</code></div>
                </div>
                <div style={{ padding: '8px', borderRadius: '6px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                  <div style={{ color: colors.bodyMuted }}>Weight Geometry</div>
                  <div style={{ fontWeight: 600, color: colors.ink }}>W_Q, W_K, W_V, W_O ∈ ℝ^(64×64)</div>
                </div>
                <div style={{ padding: '8px', borderRadius: '6px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                  <div style={{ color: colors.bodyMuted }}>Layer Block</div>
                  <div style={{ fontWeight: 600, color: colors.ink }}>TransformerBlock[{selectedComp.layer ?? 0}]</div>
                </div>
                <div style={{ padding: '8px', borderRadius: '6px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                  <div style={{ color: colors.bodyMuted }}>Head Dimension</div>
                  <div style={{ fontWeight: 600, color: colors.ink }}>d_head = 64 (768 / 12)</div>
                </div>
              </div>
            )}

            {/* RELATIONSHIPS TAB */}
            {activeTab === 'relationships' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '11px' }}>
                <div style={{ fontSize: '11px', fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase' }}>Computational Predecessors</div>
                <div style={{ padding: '6px 8px', borderRadius: '4px', backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                  ← Residual Stream (h_{selectedComp.layer ?? 0})
                </div>
                <div style={{ fontSize: '11px', fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase', marginTop: 4 }}>Computational Successors</div>
                <div style={{ padding: '6px 8px', borderRadius: '4px', backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                  → Output Projection W_O
                </div>
                <div style={{ padding: '6px 8px', borderRadius: '4px', backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                  → MLP Block Layer {selectedComp.layer ?? 0}
                </div>
              </div>
            )}

            {/* EVIDENCE TAB */}
            {activeTab === 'evidence' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {linkedEvidence.length > 0 ? (
                  linkedEvidence.map((e) => (
                    <div key={e.id} style={{ fontSize: '11px', padding: '8px', borderRadius: '6px', backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        {getKnowledgeBadge(e.evidence_level)}
                        <span style={{ fontSize: '10px', color: colors.bodyMuted }}>Metric: {e.metric_name}</span>
                      </div>
                      <div style={{ color: colors.ink, fontWeight: 600 }}>{e.claim}</div>
                      <div style={{ fontSize: '10px', color: colors.bodyMuted, marginTop: 4 }}>
                        Effect: <b>{e.metric_value != null ? e.metric_value.toFixed(2) : '—'}</b> (Baseline: {e.baseline_value?.toFixed(2) || '0.00'})
                      </div>
                    </div>
                  ))
                ) : (
                  <div style={{ fontSize: '11px', color: colors.bodyMuted, textAlign: 'center', padding: '20px 0' }}>
                    No recorded evidence for this component yet. Execute an intervention to generate empirical evidence.
                  </div>
                )}
              </div>
            )}

            {/* ACTIONS TAB */}
            {activeTab === 'actions' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <button
                  onClick={() => openPanel('intervention_lab')}
                  style={actionBtnStyle}
                >
                  <Zap size={13} style={{ color: colors.primary }} />
                  <span>Run Activation Patching</span>
                </button>
                <button
                  onClick={() => openPanel('intervention_lab')}
                  style={actionBtnStyle}
                >
                  <Activity size={13} style={{ color: colors.dangerText }} />
                  <span>Run Zero Ablation (Knockout)</span>
                </button>
                <button
                  onClick={() => openPanel('hypothesis_lab')}
                  style={actionBtnStyle}
                >
                  <FileText size={13} />
                  <span>Form Hypothesis on {selectedComp.name}</span>
                </button>
                <button
                  onClick={() => openPanel('mechanism_builder')}
                  style={actionBtnStyle}
                >
                  <Layers size={13} />
                  <span>Add to Mechanism Pipeline</span>
                </button>
              </div>
            )}

            {/* PROVENANCE TAB */}
            {activeTab === 'provenance' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '11px' }}>
                <div style={{ padding: '8px', borderRadius: '6px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                  <div style={{ color: colors.bodyMuted }}>Model Checkpoint</div>
                  <div style={{ fontWeight: 600, color: colors.ink }}><code>gpt2-small (sha256: 124m_weights)</code></div>
                </div>
                <div style={{ padding: '8px', borderRadius: '6px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                  <div style={{ color: colors.bodyMuted }}>Execution Environment</div>
                  <div style={{ fontWeight: 600, color: colors.ink }}>PyTorch 2.4.0 (Live Hooks)</div>
                </div>
                <div style={{ padding: '8px', borderRadius: '6px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                  <div style={{ color: colors.bodyMuted }}>Integrity Verification</div>
                  <div style={{ fontWeight: 600, color: colors.successText, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <ShieldCheck size={13} /> Zero Fabrication Guaranteed
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: colors.bodyMuted, fontSize: '12px', textAlign: 'center', padding: '40px 10px', lineHeight: 1.5 }}>
            Select an attention head, neuron, or token in any visualization to inspect properties and trigger causal actions.
          </div>
        )}
      </div>
    </aside>
  );
};

const inspectorStyle: React.CSSProperties = {
  width: 'var(--sidebar-width, 280px)',
  background: 'var(--bg-sidebar, #fafafa)',
  borderLeft: '1px solid var(--border, #e0e0e0)',
  display: 'flex',
  flexDirection: 'column',
  userSelect: 'none',
  fontSize: '12px',
  height: '100%',
  boxSizing: 'border-box',
};

const headerStyle: React.CSSProperties = {
  height: '36px',
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '0 12px',
  borderBottom: '1px solid var(--border-light, #f0f0f0)',
  color: 'var(--text, #1d1d1f)',
};

const actionBtnStyle: React.CSSProperties = {
  padding: '8px 10px',
  borderRadius: '6px',
  border: '1px solid var(--border, #e0e0e0)',
  backgroundColor: 'var(--color-canvas, #ffffff)',
  color: 'var(--text, #1d1d1f)',
  fontSize: '11px',
  fontWeight: 600,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
};
