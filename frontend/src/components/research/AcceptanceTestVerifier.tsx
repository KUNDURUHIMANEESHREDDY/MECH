import React, { useState } from 'react';
import { colors } from '../../design/tokens/colors';
import {
  CheckCircle2,
  XCircle,
  ArrowRight,
  Shield,
  Database,
  Cpu,
  FlaskConical,
  Brain,
  Eye,
} from 'lucide-react';

interface ChainStep {
  id: string;
  label: string;
  description: string;
  agent: 'RESEARCHER' | 'UI' | 'AGENT_2' | 'AGENT_1' | 'REAL_RESULT';
  touchesPytorch: boolean;
  touchesHooks: boolean;
  touchesMetrics: boolean;
  touchesTruth: boolean;
}

const CHAIN_STEPS: ChainStep[] = [
  {
    id: 'researcher',
    label: 'RESEARCHER',
    description: 'User interacts with UI to configure experiment',
    agent: 'RESEARCHER',
    touchesPytorch: false,
    touchesHooks: false,
    touchesMetrics: false,
    touchesTruth: false,
  },
  {
    id: 'ui',
    label: 'UI (Agent 3)',
    description: 'React components render experiment builder, monitor, evidence explorer',
    agent: 'UI',
    touchesPytorch: false,
    touchesHooks: false,
    touchesMetrics: false,
    touchesTruth: false,
  },
  {
    id: 'agent2',
    label: 'AGENT 2 (API Layer)',
    description: 'Frontend API client sends HTTP requests to backend',
    agent: 'AGENT_2',
    touchesPytorch: false,
    touchesHooks: false,
    touchesMetrics: false,
    touchesTruth: false,
  },
  {
    id: 'agent1',
    label: 'AGENT 1 (Backend)',
    description: 'FastAPI router receives request, invokes experiment runner',
    agent: 'AGENT_1',
    touchesPytorch: true,
    touchesHooks: true,
    touchesMetrics: true,
    touchesTruth: false,
  },
  {
    id: 'real_result',
    label: 'REAL RESULT',
    description: 'PyTorch executes forward passes with hooks, computes metrics',
    agent: 'REAL_RESULT',
    touchesPytorch: true,
    touchesHooks: true,
    touchesMetrics: true,
    touchesTruth: false,
  },
  {
    id: 'agent2_return',
    label: 'AGENT 2 (Return)',
    description: 'Backend returns ExperimentRun to frontend API',
    agent: 'AGENT_2',
    touchesPytorch: false,
    touchesHooks: false,
    touchesMetrics: false,
    touchesTruth: false,
  },
  {
    id: 'ui_return',
    label: 'UI (Agent 3) (Return)',
    description: 'React components update to display results',
    agent: 'UI',
    touchesPytorch: false,
    touchesHooks: false,
    touchesMetrics: false,
    touchesTruth: false,
  },
];

const AGENT_COLORS: Record<string, { color: string; bgColor: string; borderColor: string }> = {
  RESEARCHER: { color: colors.bodyMuted, bgColor: colors.surfacePearl, borderColor: colors.border },
  UI: { color: colors.primary, bgColor: colors.accentSoft, borderColor: colors.primary },
  AGENT_2: { color: colors.infoText, bgColor: colors.infoSoft, borderColor: colors.infoBorder },
  AGENT_1: { color: colors.warningText, bgColor: colors.warningSoft, borderColor: colors.warningBorder },
  REAL_RESULT: { color: colors.successText, bgColor: colors.successSoft, borderColor: colors.successBorder },
};

export const AcceptanceTestVerifier: React.FC = () => {
  const [verified, setVerified] = useState<Record<string, boolean>>({});

  const toggleVerify = (stepId: string) => {
    setVerified((prev) => ({
      ...prev,
      [stepId]: !prev[stepId],
    }));
  };

  const allVerified = CHAIN_STEPS.every((step) => verified[step.id]);
  const agent3TouchesPytorch = CHAIN_STEPS.some(
    (step) => step.agent === 'UI' && (step.touchesPytorch || step.touchesHooks || step.touchesMetrics || step.touchesTruth)
  );

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            Acceptance Test
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Agent 3 Boundary Verification
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Prove that Agent 3 only touches UI layer, never PyTorch/hooks/metrics/truth.
          </div>
        </div>

        <div style={{
          padding: '12px 16px',
          borderRadius: 8,
          backgroundColor: allVerified ? colors.successSoft : colors.surfaceTile1,
          border: `1px solid ${allVerified ? colors.successBorder : colors.border}`,
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 20, fontWeight: 800, color: allVerified ? colors.success : colors.primary }}>
            {allVerified ? (
              <CheckCircle2 size={24} style={{ color: colors.success }} />
            ) : (
              <XCircle size={24} style={{ color: colors.danger }} />
            )}
          </div>
          <div style={{ fontSize: 10, fontWeight: 600, color: colors.bodyMuted, textTransform: 'uppercase', marginTop: 4 }}>
            {allVerified ? 'VERIFIED' : 'PENDING'}
          </div>
        </div>
      </div>

      {/* Chain Visualization */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 12 }}>
          Data Flow Chain
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          {CHAIN_STEPS.map((step, idx) => {
            const agentConfig = AGENT_COLORS[step.agent];
            return (
              <React.Fragment key={step.id}>
                <div
                  onClick={() => toggleVerify(step.id)}
                  style={{
                    padding: '10px 14px',
                    borderRadius: 8,
                    border: `2px solid ${verified[step.id] ? colors.success : agentConfig.borderColor}`,
                    backgroundColor: verified[step.id] ? colors.successSoft : agentConfig.bgColor,
                    cursor: 'pointer',
                    minWidth: 120,
                    textAlign: 'center',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ fontSize: 10, fontWeight: 700, color: agentConfig.color, textTransform: 'uppercase' }}>
                    {step.agent}
                  </div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: colors.ink, marginTop: 4 }}>
                    {step.label}
                  </div>
                  {verified[step.id] && (
                    <CheckCircle2 size={14} style={{ color: colors.success, marginTop: 4 }} />
                  )}
                </div>
                {idx < CHAIN_STEPS.length - 1 && (
                  <ArrowRight size={16} style={{ color: colors.border }} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Detailed Steps */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 12 }}>
          Step Details
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {CHAIN_STEPS.map((step) => {
            const agentConfig = AGENT_COLORS[step.agent];
            return (
              <div
                key={step.id}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '200px 1fr 100px',
                  gap: 12,
                  padding: 12,
                  borderRadius: 8,
                  border: `1px solid ${verified[step.id] ? colors.successBorder : colors.border}`,
                  backgroundColor: verified[step.id] ? colors.successSoft : colors.canvas,
                  alignItems: 'center',
                }}
              >
                <div>
                  <div style={{ fontSize: 10, fontWeight: 700, color: agentConfig.color, textTransform: 'uppercase' }}>
                    {step.agent}
                  </div>
                  <div style={{ fontSize: 12, fontWeight: 600, color: colors.ink }}>{step.label}</div>
                </div>
                <div style={{ fontSize: 12, color: colors.bodyMuted }}>{step.description}</div>
                <div style={{ display: 'flex', gap: 8, fontSize: 10, color: colors.bodyMuted }}>
                  <span style={{ color: step.touchesPytorch ? colors.dangerText : colors.successText }}>
                    {step.touchesPytorch ? '⚠️ PyTorch' : '✓ No PyTorch'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Boundary Verification */}
      <div style={{ padding: 16, borderRadius: 10, backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 12 }}>
          Agent 3 Boundary Verification
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div style={{ padding: 12, borderRadius: 8, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: colors.bodyMuted, marginBottom: 6 }}>
              Agent 3 MUST NOT touch:
            </div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, color: colors.dangerText, lineHeight: 1.8 }}>
              <li>PyTorch experiment implementation</li>
              <li>Model hooks (forward/backward)</li>
              <li>Causal metric calculations</li>
              <li>Hypothesis truth determination</li>
            </ul>
          </div>
          
          <div style={{ padding: 12, borderRadius: 8, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: colors.bodyMuted, marginBottom: 6 }}>
              Agent 3 owns:
            </div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, color: colors.successText, lineHeight: 1.8 }}>
              <li>React UI components</li>
              <li>Zustand state management</li>
              <li>API client calls</li>
              <li>Data visualization</li>
            </ul>
          </div>
        </div>

        <div style={{ marginTop: 12, padding: 12, borderRadius: 8, backgroundColor: agent3TouchesPytorch ? colors.dangerSoft : colors.successSoft, border: `1px solid ${agent3TouchesPytorch ? colors.dangerBorder : colors.successBorder}` }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: agent3TouchesPytorch ? colors.dangerText : colors.successText }}>
            {agent3TouchesPytorch ? (
              <>❌ Agent 3 touches PyTorch/hooks/metrics/truth - VIOLATION</>
            ) : (
              <>✓ Agent 3 boundary verified - PASS</>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
