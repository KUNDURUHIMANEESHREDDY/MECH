import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  Sparkles,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Activity,
  Play,
  HelpCircle,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useResearchStore } from '../../shared/stores/research';

export const MechanismCriticPanel: React.FC = () => {
  const { activeInvestigation } = useResearchStore();
  const [audit, setAudit] = useState<any>({
    target_component: 'L9H9',
    epistemic_grade: 'ROBUST_CAUSAL',
    has_causal_intervention: true,
    has_negative_control: true,
    has_deterministic_replication: true,
    sample_size: 100,
    unresolved_alternatives: [
      'Alternative explanation: MLP Layer 8 feedforward contribution unresolved',
      'Alternative explanation: L8H4 backup name-mover compensation',
    ],
    limitations: ['Cross-dataset template transfer not yet evaluated'],
  });

  const [proposal, setProposal] = useState<any>({
    target_component: 'MLP_L8',
    intervention_type: 'ABLATION_ZERO',
    recommended_action: 'Ablate MLP Layer 8 while preserving attention head L9H9 and measure Δlogit',
    hypothesis_to_test_or_resolve: 'Distinguish attention name-moving from feedforward associative retrieval',
    expected_information_gain: 'HIGH',
    scientific_rationale:
      'Directly tests whether MLP L8 performs independent associative computation or routes through L9H9.',
  });

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: colors.canvas,
        color: colors.ink,
        padding: 20,
        overflowY: 'auto',
        gap: 16,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: `1px solid ${colors.hairline}`,
          paddingBottom: 14,
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ShieldAlert size={18} color={colors.primary} />
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>
              Mechanism Critic & Next-Best-Experiment Recommender
            </h2>
          </div>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 4 }}>
            Rigorous scientific evaluation of claim validity, unexamined gaps, and high-information-gain experiment proposals.
          </div>
        </div>
      </div>

      {/* Recommended Next Experiment Banner */}
      <div
        style={{
          padding: 16,
          borderRadius: 8,
          backgroundColor: colors.primarySoft,
          border: `1px solid ${colors.primary}`,
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 700, fontSize: 13, color: colors.primary }}>
            <Sparkles size={16} /> Recommended Next Best Experiment
          </div>
          <span
            style={{
              fontSize: 10,
              fontWeight: 800,
              padding: '2px 8px',
              borderRadius: 4,
              backgroundColor: colors.primary,
              color: colors.onPrimary,
            }}
          >
            INFO GAIN: {proposal.expected_information_gain}
          </span>
        </div>

        <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>
          {proposal.recommended_action}
        </div>

        <div style={{ fontSize: 11, color: colors.bodyText }}>
          <strong>Goal:</strong> {proposal.hypothesis_to_test_or_resolve}
        </div>

        <div style={{ fontSize: 11, color: colors.bodyMuted, fontStyle: 'italic' }}>
          "{proposal.scientific_rationale}"
        </div>
      </div>

      {/* Mechanism Critic Audit Card */}
      <div
        style={{
          padding: 16,
          borderRadius: 8,
          backgroundColor: colors.surfaceTile1,
          border: `1px solid ${colors.hairline}`,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
            Claim Epistemic Audit: Component {audit.target_component}
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
            GRADE: {audit.epistemic_grade}
          </span>
        </div>

        {/* Verification Checklist */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8, fontSize: 11 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <CheckCircle2 size={14} color={colors.successText} />
            <span>Causal Intervention: Executed</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <CheckCircle2 size={14} color={colors.successText} />
            <span>Negative Control: Isolated</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <CheckCircle2 size={14} color={colors.successText} />
            <span>Replication: Confirmed</span>
          </div>
        </div>

        {/* Unresolved Alternatives */}
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, marginBottom: 4 }}>
            Unresolved Competing Explanations
          </div>
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 11, color: colors.bodyMuted }}>
            {audit.unresolved_alternatives.map((alt: string, i: number) => (
              <li key={i}>{alt}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};
