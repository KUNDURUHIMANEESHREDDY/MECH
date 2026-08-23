import React, { useState } from 'react';
import {
  ListPlus,
  Play,
  CheckCircle2,
  Clock,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  AlertCircle,
  HelpCircle,
  Layers,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface QueuedExperiment {
  id: string;
  title: string;
  targetComponent: string;
  interventionType: string;
  priority: 'HIGH' | 'MEDIUM' | 'LOW';
  reason: string;
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED';
  outcomeA: string;
  outcomeB: string;
  outcomeC: string;
  infoGainScore: number;
}

export const ResearchQueueView: React.FC = () => {
  const [queue, setQueue] = useState<QueuedExperiment[]>([
    {
      id: 'q1',
      title: 'Ablate MLP_L8 under Matched IOI Conditions',
      targetComponent: 'MLP_L8',
      interventionType: 'ZERO_ABLATION',
      priority: 'HIGH',
      reason: 'Current evidence establishes L9H9 causality, but does not distinguish whether MLP8 independently mediates the name-copying behavior.',
      status: 'QUEUED',
      outcomeA: 'MLP8 has zero effect (Δ < 0.10) → Confirms attention-only pathway for H1.',
      outcomeB: 'MLP8 has comparable effect (Δ > 1.20) → Falsifies pure-attention hypothesis in favor of parallel mediation.',
      outcomeC: 'Both interact non-linearly → Indicates compound multi-component mechanism.',
      infoGainScore: 0.94,
    },
    {
      id: 'q2',
      title: 'L8H2 Contrastive Neighboring-Head Intervention',
      targetComponent: 'L8H2',
      interventionType: 'ACTIVATION_PATCHING',
      priority: 'MEDIUM',
      reason: 'Validates negative control specificity by testing an adjacent head within the same layer.',
      status: 'QUEUED',
      outcomeA: 'Zero effect → Confirms tight spatial localization to L8H1.',
      outcomeB: 'Significant effect → Broadens duplicate-token cluster to adjacent heads.',
      outcomeC: 'Partial effect → Points to distributed head ensemble.',
      infoGainScore: 0.78,
    },
    {
      id: 'q3',
      title: 'Cross-Dataset Name-Swap Generalization (BABA Templates)',
      targetComponent: 'L9H9',
      interventionType: 'ZERO_ABLATION',
      priority: 'HIGH',
      reason: 'Tests whether L9H9 causal effect generalizes across inverted syntactical constructions.',
      status: 'QUEUED',
      outcomeA: 'Effect holds across 50 templates → Establishes cross-dataset robustness.',
      outcomeB: 'Effect drops significantly → Indicates template-specific memorization.',
      outcomeC: 'Inverted effect → Discovers syntactic polarity sensitivity.',
      infoGainScore: 0.88,
    },
  ]);

  const [isRunningAll, setIsRunningAll] = useState(false);

  const handleRunAll = () => {
    setIsRunningAll(true);
    // Simulate background runner dispatching to job queue
    setTimeout(() => {
      setQueue((prev) =>
        prev.map((item) => ({ ...item, status: 'COMPLETED' }))
      );
      setIsRunningAll(false);
    }, 1800);
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
            <Sparkles size={18} color={colors.primary} />
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>
              What Should I Test Next? — Research Queue
            </h2>
          </div>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 4 }}>
            Critic-recommended discriminative experiments ordered by expected information gain. Queue and execute via the background runner.
          </div>
        </div>

        <button
          onClick={handleRunAll}
          disabled={isRunningAll}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            backgroundColor: isRunningAll ? colors.surfaceTile2 : colors.primary,
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            padding: '8px 14px',
            fontSize: 12,
            fontWeight: 700,
            cursor: isRunningAll ? 'default' : 'pointer',
          }}
        >
          <Play size={14} />
          {isRunningAll ? 'Running Queue...' : 'Run All Queued Experiments'}
        </button>
      </div>

      {/* Experiment Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {queue.map((item, index) => (
          <div
            key={item.id}
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
            {/* Top Bar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 700,
                      padding: '2px 6px',
                      borderRadius: 4,
                      backgroundColor: item.priority === 'HIGH' ? colors.warningSoft || colors.surfaceTile2 : colors.surfaceTile2,
                      color: item.priority === 'HIGH' ? colors.warningText || colors.primary : colors.bodyMuted,
                    }}
                  >
                    {item.priority} PRIORITY
                  </span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
                    {index + 1}. {item.title}
                  </span>
                </div>
                <div style={{ fontSize: 12, color: colors.bodyText, marginTop: 6 }}>
                  <strong>Scientific Rationale:</strong> {item.reason}
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '3px 8px',
                    borderRadius: 4,
                    backgroundColor: item.status === 'COMPLETED' ? colors.successSoft : colors.surfaceTile2,
                    color: item.status === 'COMPLETED' ? colors.successText : colors.bodyMuted,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                  }}
                >
                  {item.status === 'COMPLETED' ? <CheckCircle2 size={12} /> : <Clock size={12} />}
                  {item.status}
                </span>
              </div>
            </div>

            {/* Expected Outcomes Matrix */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                gap: 8,
                backgroundColor: colors.surfaceTile2,
                padding: 10,
                borderRadius: 6,
                fontSize: 11,
              }}
            >
              <div>
                <strong style={{ color: colors.successText }}>Outcome A:</strong>{' '}
                <span style={{ color: colors.bodyMuted }}>{item.outcomeA}</span>
              </div>
              <div>
                <strong style={{ color: colors.warningText || colors.primary }}>Outcome B:</strong>{' '}
                <span style={{ color: colors.bodyMuted }}>{item.outcomeB}</span>
              </div>
              <div>
                <strong style={{ color: colors.primary }}>Outcome C:</strong>{' '}
                <span style={{ color: colors.bodyMuted }}>{item.outcomeC}</span>
              </div>
            </div>

            {/* Footer Bar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11 }}>
              <div style={{ color: colors.bodyMuted }}>
                Target: <strong style={{ fontFamily: 'monospace', color: colors.primary }}>{item.targetComponent}</strong> · Intervention: <strong>{item.interventionType}</strong>
              </div>
              <div style={{ color: colors.bodyMuted }}>
                Expected Info Gain: <strong style={{ color: colors.ink }}>{item.infoGainScore * 100}%</strong>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
