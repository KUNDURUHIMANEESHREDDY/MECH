import React, { useState } from 'react';
import {
  GitCompare,
  ArrowRight,
  CheckCircle2,
  XCircle,
  Layers,
  FlaskConical,
  Scale,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';

export const ComparisonWorkspaceView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'EXPERIMENTS' | 'MECHANISMS'>('EXPERIMENTS');

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
      {/* Header & Tabs */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: `1px solid ${colors.hairline}`,
          paddingBottom: 14,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Scale size={18} color={colors.primary} />
          <div>
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>
              Scientific Comparison Workspace
            </h2>
            <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
              Differential analysis comparing experimental runs and competing circuit mechanisms.
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, backgroundColor: colors.surfaceTile1, padding: 3, borderRadius: 6 }}>
          <button
            onClick={() => setActiveTab('EXPERIMENTS')}
            style={{
              padding: '6px 12px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 700,
              border: 'none',
              cursor: 'pointer',
              backgroundColor: activeTab === 'EXPERIMENTS' ? colors.primary : 'transparent',
              color: activeTab === 'EXPERIMENTS' ? '#fff' : colors.bodyMuted,
            }}
          >
            Experiment Comparison
          </button>
          <button
            onClick={() => setActiveTab('MECHANISMS')}
            style={{
              padding: '6px 12px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 700,
              border: 'none',
              cursor: 'pointer',
              backgroundColor: activeTab === 'MECHANISMS' ? colors.primary : 'transparent',
              color: activeTab === 'MECHANISMS' ? '#fff' : colors.bodyMuted,
            }}
          >
            Mechanism Comparison
          </button>
        </div>
      </div>

      {/* Comparison Grid */}
      {activeTab === 'EXPERIMENTS' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Experiment A */}
            <div
              style={{
                backgroundColor: colors.surfaceTile1,
                border: `1px solid ${colors.hairline}`,
                borderRadius: 8,
                padding: 16,
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
                  Experiment A: L9H9 Zero-Ablation
                </span>
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
                  TREATMENT RUN
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11 }}>
                <div><strong>Model:</strong> <span style={{ color: colors.bodyMuted }}>gpt2 (124M)</span></div>
                <div><strong>Dataset:</strong> <span style={{ color: colors.bodyMuted }}>IOI Clean Name-Swap Prompts</span></div>
                <div><strong>Intervention:</strong> <span style={{ color: colors.primary }}>Zero Ablation (L9H9)</span></div>
                <div><strong>Negative Control:</strong> <span style={{ color: colors.bodyMuted }}>L0H0 (Baseline Isolated)</span></div>
                <div><strong>Treatment Effect:</strong> <strong style={{ color: colors.successText }}>Δlogit = +2.13</strong> [COMPUTED]</div>
                <div><strong>Replications:</strong> <span style={{ color: colors.bodyMuted }}>3/3 Verified</span></div>
              </div>
            </div>

            {/* Experiment B */}
            <div
              style={{
                backgroundColor: colors.surfaceTile1,
                border: `1px solid ${colors.hairline}`,
                borderRadius: 8,
                padding: 16,
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
                  Experiment B: L0H0 Control Ablation
                </span>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 4,
                    backgroundColor: colors.surfaceTile2,
                    color: colors.bodyMuted,
                  }}
                >
                  CONTROL BASELINE
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11 }}>
                <div><strong>Model:</strong> <span style={{ color: colors.bodyMuted }}>gpt2 (124M)</span></div>
                <div><strong>Dataset:</strong> <span style={{ color: colors.bodyMuted }}>IOI Clean Name-Swap Prompts</span></div>
                <div><strong>Intervention:</strong> <span style={{ color: colors.bodyMuted }}>Zero Ablation (L0H0)</span></div>
                <div><strong>Negative Control:</strong> <span style={{ color: colors.bodyMuted }}>Self</span></div>
                <div><strong>Treatment Effect:</strong> <strong style={{ color: colors.bodyMuted }}>Δlogit = +0.11</strong> [COMPUTED]</div>
                <div><strong>Replications:</strong> <span style={{ color: colors.bodyMuted }}>3/3 Verified</span></div>
              </div>
            </div>
          </div>

          {/* Differential Conclusion */}
          <div
            style={{
              backgroundColor: colors.surfaceTile2,
              padding: 14,
              borderRadius: 6,
              fontSize: 12,
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
            }}
          >
            <strong style={{ color: colors.ink }}>Differential Analysis:</strong>
            <div style={{ color: colors.bodyText }}>
              L9H9 zero ablation produces a statistically significant <strong style={{ color: colors.successText }}>+2.02 logit advantage</strong> over the L0H0 control baseline (Cohen's d = 3.42), proving specific causal mediation over non-specific degradation.
            </div>
          </div>
        </div>
      ) : (
        /* Mechanism Comparison Mode */
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Mechanism A */}
            <div
              style={{
                backgroundColor: colors.surfaceTile1,
                border: `1px solid ${colors.hairline}`,
                borderRadius: 8,
                padding: 16,
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
                Mechanism Alpha: L8H1 ➔ L9H9 ➔ Residual Stream
              </div>
              <div style={{ fontSize: 11, display: 'flex', flexDirection: 'column', gap: 4 }}>
                <div><strong>Status:</strong> <span style={{ color: colors.successText }}>CAUSAL MATCH</span></div>
                <div><strong>Causal Interventions:</strong> 3 validated</div>
                <div><strong>Negative Controls:</strong> Isolated (L0H0)</div>
                <div><strong>Replication:</strong> 3 independent runs</div>
              </div>
            </div>

            {/* Mechanism B */}
            <div
              style={{
                backgroundColor: colors.surfaceTile1,
                border: `1px solid ${colors.hairline}`,
                borderRadius: 8,
                padding: 16,
                display: 'flex',
                flexDirection: 'column',
                gap: 10,
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
                Mechanism Beta: L8H1 ➔ MLP_L8 ➔ Residual Stream
              </div>
              <div style={{ fontSize: 11, display: 'flex', flexDirection: 'column', gap: 4 }}>
                <div><strong>Status:</strong> <span style={{ color: colors.warningText || colors.primary }}>CANDIDATE (Unresolved)</span></div>
                <div><strong>Causal Interventions:</strong> 1 (Uncontrolled)</div>
                <div><strong>Negative Controls:</strong> Missing</div>
                <div><strong>Replication:</strong> Incomplete</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
