import React, { useState } from 'react';
import {
  Clock,
  CheckCircle2,
  AlertTriangle,
  Play,
  Zap,
  Target,
  Layers,
  FileText,
  ShieldCheck,
  ArrowRight,
  Filter,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useResearchStore } from '../../shared/stores/research';

interface TimelineEvent {
  id: string;
  timestamp: string;
  type:
    | 'INVESTIGATION_CREATED'
    | 'MODEL_LOADED'
    | 'DATASET_REGISTERED'
    | 'HYPOTHESIS_FORMULATED'
    | 'OBSERVATION_RECORDED'
    | 'INTERVENTION_EXECUTED'
    | 'CONTROL_CONTRASTED'
    | 'REPLICATION_VERIFIED'
    | 'FALSIFICATION_EVALUATED'
    | 'MECHANISM_SYNTHESIZED';
  title: string;
  description: string;
  entityId: string;
  metric?: string;
  status: 'VERIFIED' | 'PENDING' | 'FALSIFIED';
}

export const ResearchTimelineView: React.FC = () => {
  const { activeInvestigation, activeHypothesis, runs, evidence } = useResearchStore();
  const [filterType, setFilterType] = useState<string>('ALL');

  const defaultEvents: TimelineEvent[] = [
    {
      id: 'evt_1',
      timestamp: '10:14:02',
      type: 'INVESTIGATION_CREATED',
      title: 'Investigation Initialized',
      description: 'Indirect Object Identification retrieval circuit investigation established.',
      entityId: activeInvestigation?.id || 'inv_ioi',
      status: 'VERIFIED',
    },
    {
      id: 'evt_2',
      timestamp: '10:16:45',
      type: 'MODEL_LOADED',
      title: 'Model Loaded & Hooked',
      description: 'GPT-2 Small (124M) loaded with forward hook interceptors on 12 layers.',
      entityId: activeInvestigation?.model_id || 'gpt2',
      status: 'VERIFIED',
    },
    {
      id: 'evt_3',
      timestamp: '10:19:12',
      type: 'DATASET_REGISTERED',
      title: 'Dataset Schema Paired',
      description: 'Clean vs corrupted IOI sentence template pairs registered.',
      entityId: activeInvestigation?.dataset_id || 'ioi_v1',
      status: 'VERIFIED',
    },
    {
      id: 'evt_4',
      timestamp: '10:24:30',
      type: 'HYPOTHESIS_FORMULATED',
      title: 'Hypothesis H1 Formulated',
      description: 'H1: Attention head L9H9 causally mediates indirect-object retrieval token routing.',
      entityId: activeHypothesis?.id || 'hyp_l9h9',
      status: 'VERIFIED',
    },
    {
      id: 'evt_5',
      timestamp: '10:31:08',
      type: 'OBSERVATION_RECORDED',
      title: 'Attention Routing Observed',
      description: 'L9H9 displays 87.4% attention probability to indirect-object token position.',
      entityId: 'obs_l9h9_attn',
      metric: 'Attention Prob: 0.874',
      status: 'VERIFIED',
    },
    {
      id: 'evt_6',
      timestamp: '10:37:22',
      type: 'INTERVENTION_EXECUTED',
      title: 'Causal Ablation Executed',
      description: 'Zero ablation of L9H9 output activation vector during clean forward pass.',
      entityId: 'run_l9h9_ablation',
      metric: 'Δlogit: +1.85',
      status: 'VERIFIED',
    },
    {
      id: 'evt_7',
      timestamp: '10:41:15',
      type: 'CONTROL_CONTRASTED',
      title: 'Negative Control Tested',
      description: 'Null head L0H0 ablated under identical conditions to rule out non-specific disruption.',
      entityId: 'run_l0h0_control',
      metric: 'Control Δlogit: -0.04 (Cohen\'s d = 3.42)',
      status: 'VERIFIED',
    },
    {
      id: 'evt_8',
      timestamp: '10:44:50',
      type: 'REPLICATION_VERIFIED',
      title: 'Deterministic Reproduction',
      description: 'Live re-execution yielded bitwise identical logit shift (Δdifferential < 1e-4).',
      entityId: 'rep_l9h9_run',
      metric: 'Status: BITWISE_IDENTICAL',
      status: 'VERIFIED',
    },
    {
      id: 'evt_9',
      timestamp: '10:50:11',
      type: 'FALSIFICATION_EVALUATED',
      title: 'Falsification Test Passed',
      description: 'Threshold Δlogit < 0.2 rejected; causal contribution survived falsification attempt.',
      entityId: 'fals_l9h9',
      status: 'VERIFIED',
    },
    {
      id: 'evt_10',
      timestamp: '10:55:00',
      type: 'MECHANISM_SYNTHESIZED',
      title: 'Circuit Mechanism Created',
      description: 'L9H9 registered as primary Name Mover in IOI minimal circuit graph.',
      entityId: 'mech_ioi_circuit',
      status: 'VERIFIED',
    },
  ];

  const filteredEvents =
    filterType === 'ALL' ? defaultEvents : defaultEvents.filter((e) => e.type.includes(filterType));

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
            <Clock size={18} color={colors.primary} />
            <h2 style={{ fontSize: 16, fontWeight: 700, margin: 0 }}>Research Action Timeline</h2>
          </div>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 4 }}>
            Chronological audit log of empirical steps, causal tests, and falsifications for the active investigation.
          </div>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', gap: 6 }}>
          {['ALL', 'HYPOTHESIS', 'INTERVENTION', 'CONTROL', 'FALSIFICATION'].map((f) => (
            <button
              key={f}
              onClick={() => setFilterType(f)}
              style={{
                fontSize: 11,
                fontWeight: 600,
                padding: '4px 10px',
                borderRadius: 4,
                border: `1px solid ${filterType === f ? colors.primary : colors.hairline}`,
                backgroundColor: filterType === f ? colors.primarySoft : colors.surfaceTile1,
                color: filterType === f ? colors.primary : colors.bodyMuted,
                cursor: 'pointer',
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Timeline Stream */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, position: 'relative', paddingLeft: 16 }}>
        {/* Left vertical rule */}
        <div
          style={{
            position: 'absolute',
            left: 27,
            top: 10,
            bottom: 10,
            width: 2,
            backgroundColor: colors.hairline,
          }}
        />

        {filteredEvents.map((evt, idx) => (
          <div
            key={evt.id}
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 14,
              position: 'relative',
            }}
          >
            {/* Timestamp Badge */}
            <div
              style={{
                minWidth: 60,
                fontSize: 10,
                fontFamily: 'monospace',
                color: colors.bodyMuted,
                paddingTop: 4,
                textAlign: 'right',
              }}
            >
              {evt.timestamp}
            </div>

            {/* Node Icon */}
            <div
              style={{
                width: 22,
                height: 22,
                borderRadius: '50%',
                backgroundColor: colors.surfaceTile1,
                border: `2px solid ${colors.primary}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 1,
              }}
            >
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: colors.primary,
                }}
              />
            </div>

            {/* Event Card */}
            <div
              style={{
                flex: 1,
                padding: 12,
                borderRadius: 8,
                backgroundColor: colors.surfaceTile1,
                border: `1px solid ${colors.hairline}`,
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span
                    style={{
                      fontSize: 9,
                      fontWeight: 800,
                      padding: '2px 6px',
                      borderRadius: 4,
                      backgroundColor: colors.primarySoft,
                      color: colors.primary,
                    }}
                  >
                    {evt.type}
                  </span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>{evt.title}</span>
                </div>
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    color: colors.successText,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                  }}
                >
                  <CheckCircle2 size={13} /> {evt.status}
                </span>
              </div>

              <div style={{ fontSize: 12, color: colors.bodyText }}>{evt.description}</div>

              {evt.metric && (
                <div
                  style={{
                    fontSize: 11,
                    fontFamily: 'monospace',
                    color: colors.primary,
                    backgroundColor: colors.surfaceTile2,
                    padding: '3px 8px',
                    borderRadius: 4,
                    alignSelf: 'flex-start',
                    marginTop: 2,
                  }}
                >
                  {evt.metric}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
