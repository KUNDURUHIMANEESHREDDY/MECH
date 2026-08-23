import React from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import { colors } from '../../design/tokens/colors';
import {
  Zap,
  ShieldCheck,
  Activity,
  FileText,
  Sparkles,
} from 'lucide-react';

export const KnowledgeTypeBadge: React.FC<{type: string}> = ({type}) => {
  const typeMap: Record<string, {label: string; color: string; bg: string}> = {
    OBSERVATION: {label: 'OBSERVED', color: colors.bodyMuted, bg: colors.surfacePearl},
    INFERENCE: {label: 'INFERENCE', color: colors.bodyMuted, bg: colors.warningSoft},
    CAUSAL_EVIDENCE: {label: 'CAUSAL EVIDENCE', color: colors.successText, bg: colors.successSoft},
    CLAIM: {label: 'CLAIM', color: colors.bodyMuted, bg: colors.primarySoft},
  };
  const info = typeMap[type] || {label: type || '—', color: colors.bodyMuted, bg: colors.surfacePearl};
  return (
    <span
      style={{
        fontSize: 9,
        fontWeight: 600,
        padding: '1px 5px',
        borderRadius: 3,
        color: info.color,
        backgroundColor: info.bg,
        marginRight: 2,
      }}
    >
      {info.label}
    </span>
  );
};

export const InvestigationHeader: React.FC = () => {
  const {
    activeInvestigation,
    activeHypothesis,
    runs,
    evidence,
    knowledgeTypes,
    evidenceStatus,
    lastRun,
    reproducibility,
  } = useResearchStore();

  const { openPanel, setVisiblePanels } = useWorkspaceStore();

  const latestRun = runs.length > 0 ? runs[0] : null;

  // Preset Layout Switcher
  const applyPreset = (preset: 'investigation' | 'attention' | 'circuit' | 'patching' | 'model' | 'comparison') => {
    switch (preset) {
      case 'investigation':
        setVisiblePanels({
          active_investigation: true,
          hypothesis_lab: true,
          report_mode: false,
          attention_heatmap: false,
          neuron_panel: false,
          logit_lens: false,
          circuit_explorer: false,
          dataset_viewer: false,
          sae_feature: false,
          intervention_lab: false,
          evidence_graph: false,
          mechanism_builder: false,
          model_explorer: false,
          compute_center: false,
        });
        break;
      case 'attention':
        setVisiblePanels({
          active_investigation: false,
          attention_heatmap: true,
          logit_lens: true,
          sae_feature: true,
          hypothesis_lab: false,
          intervention_lab: false,
          evidence_graph: false,
          mechanism_builder: false,
          model_explorer: false,
          compute_center: false,
        });
        break;
      case 'circuit':
        setVisiblePanels({
          active_investigation: false,
          model_explorer: true,
          intervention_lab: true,
          mechanism_builder: true,
          evidence_graph: true,
          attention_heatmap: false,
          neuron_panel: false,
          logit_lens: false,
          circuit_explorer: false,
          dataset_viewer: false,
          sae_feature: false,
          hypothesis_lab: false,
          compute_center: false,
        });
        break;
      case 'patching':
        setVisiblePanels({
          active_investigation: false,
          intervention_lab: true,
          evidence_graph: true,
          hypothesis_lab: true,
          attention_heatmap: false,
          neuron_panel: false,
          logit_lens: false,
          circuit_explorer: false,
          dataset_viewer: false,
          sae_feature: false,
          mechanism_builder: false,
        });
        break;
      case 'model':
        setVisiblePanels({
          active_investigation: false,
          model_explorer: true,
          neuron_panel: true,
          sae_feature: true,
          attention_heatmap: false,
          logit_lens: false,
          circuit_explorer: false,
          dataset_viewer: false,
          hypothesis_lab: false,
          intervention_lab: false,
          evidence_graph: false,
        });
        break;
      case 'comparison':
        setVisiblePanels({
          active_investigation: true,
          comparison_workspace: true,
          research_queue: true,
          hypothesis_lab: false,
          attention_heatmap: false,
          neuron_panel: false,
          logit_lens: false,
          circuit_explorer: false,
          dataset_viewer: false,
          sae_feature: false,
          intervention_lab: false,
          evidence_graph: false,
          mechanism_builder: false,
          model_explorer: false,
          compute_center: false,
        });
        break;
    }
  };

  return (
    <div
      style={{
        backgroundColor: colors.surfaceTile1,
        borderBottom: `1px solid ${colors.border}`,
        padding: '8px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        userSelect: 'none',
        zIndex: 10,
      }}
    >
      {/* Top Line: Investigation Meta + Presets */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span
              style={{
                fontSize: 10,
                fontWeight: 800,
                color: colors.primary,
                backgroundColor: colors.primarySoft,
                padding: '2px 6px',
                borderRadius: 4,
                textTransform: 'uppercase',
                letterSpacing: 0.5,
              }}
            >
              INVESTIGATION
            </span>
            <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
              {activeInvestigation?.title || 'Indirect Object Identification (IOI) Circuit'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: colors.bodyMuted }}>
            <span>•</span>
            <span>Model: <b style={{ color: colors.body }}>{activeInvestigation?.model_id || 'gpt2'}</b></span>
            <span>•</span>
            <span>Dataset: <b style={{ color: colors.body }}>{activeInvestigation?.dataset_id || 'ioi'}</b></span>
            <span>•</span>
            <span>Research Question: {activeInvestigation?.research_question || '—'}</span>
          </div>
        </div>

        {/* Workspace Presets */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase', marginRight: 4 }}>
            Presets:
          </span>
          <button
            onClick={() => applyPreset('investigation')}
            style={{
              padding: '3px 7px',
              borderRadius: 4,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.body,
              fontSize: 10,
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Investigation Overview & Hypotheses"
          >
            Overview
          </button>
          <button
            onClick={() => applyPreset('attention')}
            style={{
              padding: '3px 7px',
              borderRadius: 4,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.body,
              fontSize: 10,
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Attention Lab & Logit Lens"
          >
            Attention
          </button>
          <button
            onClick={() => applyPreset('circuit')}
            style={{
              padding: '3px 7px',
              borderRadius: 4,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.body,
              fontSize: 10,
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Circuit Discovery & Mechanism Builder"
          >
            Circuit
          </button>
          <button
            onClick={() => applyPreset('patching')}
            style={{
              padding: '3px 7px',
              borderRadius: 4,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.body,
              fontSize: 10,
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Causal Patching & Evidence Graph"
          >
            Patching
          </button>
          <button
            onClick={() => applyPreset('model')}
            style={{
              padding: '3px 7px',
              borderRadius: 4,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.body,
              fontSize: 10,
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Model Explorer & Neurons"
          >
            Model
          </button>
          <button
            onClick={() => applyPreset('comparison')}
            style={{
              padding: '3px 7px',
              borderRadius: 4,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.body,
              fontSize: 10,
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Experiment Comparison"
          >
            Compare
          </button>
        </div>
      </div>

      {/* Middle Line: Hypothesis + Knowledge Type + Evidence Status */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontWeight: 600, color: colors.bodyMuted }}>Hypothesis:</span>
            <span style={{ fontWeight: 700, color: colors.ink }}>
              {activeHypothesis?.title || 'L9H9 Name Mover Hypothesis'}
            </span>
            <code
              style={{
                fontSize: 10,
                backgroundColor: colors.surfacePearl,
                padding: '1px 5px',
                borderRadius: 3,
              }}
            >
              {activeHypothesis?.target_component || 'L9H9'}
            </code>
            {knowledgeTypes?.length > 0 && (
              <span style={{ marginLeft: 8 }}>
                <b>Knowledge:</b>
                {knowledgeTypes.map((kt: string, i: number) => (
                  <KnowledgeTypeBadge key={i} type={kt} />
                ))}
              </span>
            )}
          </div>

          <span style={{ color: colors.borderLight }}>|</span>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: colors.body }}>
            <span>Experiments: <b>{runs.length}</b></span>
            {evidence?.length > 0 && (
              <span>Evidence: <b>{evidence.length}</b></span>
            )}
          </div>
        </div>

        {/* Evidence + Reproducibility Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: colors.successText, fontSize: 10, fontWeight: 700 }}>
            <ShieldCheck size={13} />
            <span>PROVENANCE: VERIFIED</span>
          </div>

          <KnowledgeTypeBadge type={evidenceStatus} />{' '}

          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: colors.bodyMuted, fontSize: 10 }}>
            <Activity size={12} />
            {lastRun?.timestamp ? (
              <span>Last run: {lastRun.timestamp}</span>
            ) : (
              <span>No runs</span>
            )}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: colors.bodyMuted, fontSize: 10 }}>
            <ShieldCheck size={12} />
            {reproducibility?.is_reproducible
              ? 'Reproducible ✓'
              : 'Not reproducible'}
          </div>
        </div>
      </div>

      {/* Bottom Line: Action Shortcuts */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button
          onClick={() => openPanel('intervention_lab')}
          style={{
            padding: '3px 8px',
            borderRadius: 4,
            border: 'none',
            backgroundColor: colors.primary,
            color: colors.onPrimary,
            fontSize: 11,
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <Zap size={11} /> Causal Patch
        </button>

        <button
          onClick={() => openPanel('report_mode')}
          style={{
            padding: '3px 8px',
            borderRadius: 4,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.canvas,
            color: colors.ink,
            fontSize: 11,
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <FileText size={11} /> Report
        </button>

        <button
          onClick={() => openPanel('evidence_graph')}
          style={{
            padding: '3px 8px',
            borderRadius: 4,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.canvas,
            color: colors.ink,
            fontSize: 11,
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <Sparkles size={11} /> Evidence
        </button>
      </div>
    </div>
  );
};