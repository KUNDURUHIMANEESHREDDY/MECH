import React, { useState } from 'react';
import {
  HelpCircle,
  CheckCircle2,
  AlertCircle,
  HelpCircle as QuestionIcon,
  XCircle,
  ChevronRight,
  ShieldCheck,
  Zap,
  Layers,
  Activity,
  FileText,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { ClaimDerivationModal } from './ClaimDerivationModal';

interface EpistemicItem {
  id: string;
  category: 'ESTABLISHED' | 'SUPPORTED' | 'CANDIDATE' | 'UNKNOWN' | 'CONTRADICTED';
  title: string;
  statement: string;
  component: string;
  evidenceSummary: string;
  provenance: string;
}

export const EpistemicStateView: React.FC = () => {
  const [selectedClaim, setSelectedClaim] = useState<EpistemicItem | null>(null);

  const items: EpistemicItem[] = [
    {
      id: 'ep_1',
      category: 'ESTABLISHED',
      title: 'Direct Attention Routing to Indirect Object',
      statement: 'Head L9H9 places 87.4% attention on the indirect-object token position during clean IOI forward pass.',
      component: 'L9H9 (Layer 9 Head 9)',
      evidenceSummary: 'Direct forward pass observation over 100 IOI prompt templates.',
      provenance: 'Observed Tensor Hook (GPT-2 Small)',
    },
    {
      id: 'ep_2',
      category: 'SUPPORTED',
      title: 'Causal Name-Mover Logit Mediation',
      statement: 'Zero/mean ablation of L9H9 drops target indirect-object logit by Δlogit = +1.85 with Cohen\'s d = 3.42 over negative controls.',
      component: 'L9H9 (Layer 9 Head 9)',
      evidenceSummary: 'Causal ablation verified with negative control (L0H0) and deterministic replication.',
      provenance: 'Causal Intervention Runner (Run run_l9h9_ablation)',
    },
    {
      id: 'ep_3',
      category: 'CANDIDATE',
      title: 'Backup Name-Mover Compensation',
      statement: 'Head L8H4 exhibits weak attention (14.2%) and may serve as a secondary redundant path when L9H9 is ablated.',
      component: 'L8H4 (Layer 8 Head 4)',
      evidenceSummary: 'Observational correlation; causal compensation test pending.',
      provenance: 'Candidate Hypothesis H2',
    },
    {
      id: 'ep_4',
      category: 'UNKNOWN',
      title: 'Feedforward Layer 8 Direct Computation',
      statement: 'Whether MLP Layer 8 directly processes associative memory independently of attention heads remains untested.',
      component: 'MLP_L8',
      evidenceSummary: 'No direct MLP knockouts executed in current campaign.',
      provenance: 'Unexamined Alternative Explanation',
    },
    {
      id: 'ep_5',
      category: 'CONTRADICTED',
      title: 'Early Attention Alone Sufficient',
      statement: 'The claim that early layers (L0-L4) fully resolve IOI without later Name Movers was falsified by ablation.',
      component: 'L0-L4 Subgraph',
      evidenceSummary: 'Ablation of L9H9 collapsed target prediction regardless of early layer fidelity.',
      provenance: 'Falsification Test (FALSIFIED)',
    },
  ];

  const categoryConfig: Record<string, { label: string; color: string; bg: string; icon: any }> = {
    ESTABLISHED: { label: 'Established Facts', color: colors.successText, bg: colors.successSoft, icon: CheckCircle2 },
    SUPPORTED: { label: 'Causally Supported', color: colors.primary, bg: colors.primarySoft, icon: ShieldCheck },
    CANDIDATE: { label: 'Candidate Hypotheses', color: colors.warningText || '#eab308', bg: colors.surfaceTile2, icon: AlertCircle },
    UNKNOWN: { label: 'Unknowns & Unexamined', color: colors.bodyMuted, bg: colors.surfaceTile1, icon: QuestionIcon },
    CONTRADICTED: { label: 'Contradicted / Falsified', color: colors.dangerText || '#ef4444', bg: colors.dangerSoft || colors.surfaceTile1, icon: XCircle },
  };

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
            <HelpCircle size={18} color={colors.primary} />
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>
              What Do We Actually Know?
            </h2>
          </div>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 4 }}>
            Epistemic breakdown distinguishing established tensor facts, causally supported claims, candidates, and falsified hypotheses.
          </div>
        </div>
      </div>

      {/* Grid of Epistemic Columns */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 14 }}>
        {(['ESTABLISHED', 'SUPPORTED', 'CANDIDATE', 'UNKNOWN', 'CONTRADICTED'] as const).map((cat) => {
          const cfg = categoryConfig[cat];
          const catItems = items.filter((i) => i.category === cat);
          const Icon = cfg.icon;

          return (
            <div
              key={cat}
              style={{
                borderRadius: 8,
                backgroundColor: colors.surfaceTile1,
                border: `1px solid ${colors.hairline}`,
                padding: 14,
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
              }}
            >
              {/* Category Header */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Icon size={16} color={cfg.color} />
                  <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>{cfg.label}</span>
                </div>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 4,
                    backgroundColor: cfg.bg,
                    color: cfg.color,
                  }}
                >
                  {catItems.length}
                </span>
              </div>

              {/* Items List */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {catItems.map((item) => (
                  <div
                    key={item.id}
                    style={{
                      padding: 10,
                      borderRadius: 6,
                      backgroundColor: colors.surfaceTile2,
                      border: `1px solid ${colors.hairline}`,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 6,
                    }}
                  >
                    <div style={{ fontWeight: 600, fontSize: 12, color: colors.ink }}>{item.title}</div>
                    <div style={{ fontSize: 11, color: colors.bodyText, lineHeight: 1.4 }}>{item.statement}</div>

                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginTop: 4,
                        paddingTop: 6,
                        borderTop: `1px dashed ${colors.hairline}`,
                      }}
                    >
                      <span style={{ fontSize: 10, fontFamily: 'monospace', color: colors.primary }}>
                        {item.component}
                      </span>

                      {cat === 'SUPPORTED' && (
                        <button
                          onClick={() => setSelectedClaim(item)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 4,
                            padding: '2px 8px',
                            fontSize: 10,
                            fontWeight: 600,
                            borderRadius: 4,
                            border: `1px solid ${colors.hairline}`,
                            backgroundColor: colors.surfaceTile1,
                            color: colors.ink,
                            cursor: 'pointer',
                          }}
                        >
                          Why is this valid? <ChevronRight size={12} />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Interactive Modal */}
      {selectedClaim && (
        <ClaimDerivationModal
          claimTitle={selectedClaim.statement}
          targetComponent={selectedClaim.component}
          onClose={() => setSelectedClaim(null)}
        />
      )}
    </div>
  );
};
