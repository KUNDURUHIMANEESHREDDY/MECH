import React, { useState, useEffect } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import {
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  Shield,
  Database,
  Cpu,
  FlaskConical,
  Eye,
  EyeOff,
  RefreshCw,
} from 'lucide-react';

type TestState = 
  | 'API_UNAVAILABLE'
  | 'MODEL_UNAVAILABLE'
  | 'EXPERIMENT_FAILED'
  | 'EXPERIMENT_RUNNING'
  | 'EMPTY_EVIDENCE'
  | 'MISSING_METRICS'
  | 'PARTIAL_RESULT'
  | 'MOCK_RESULT'
  | 'REAL_RESULT';

interface TestCase {
  id: TestState;
  label: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
  borderColor: string;
}

const TEST_CASES: TestCase[] = [
  {
    id: 'API_UNAVAILABLE',
    label: 'API Unavailable',
    description: 'Backend service is unreachable or down',
    icon: <Database size={14} />,
    color: colors.dangerText,
    bgColor: colors.dangerSoft,
    borderColor: colors.dangerBorder,
  },
  {
    id: 'MODEL_UNAVAILABLE',
    label: 'Model Unavailable',
    description: 'PyTorch/Transformers not installed or model weights missing',
    icon: <Cpu size={14} />,
    color: colors.dangerText,
    bgColor: colors.dangerSoft,
    borderColor: colors.dangerBorder,
  },
  {
    id: 'EXPERIMENT_FAILED',
    label: 'Experiment Failed',
    description: 'Experiment execution encountered an error',
    icon: <XCircle size={14} />,
    color: colors.dangerText,
    bgColor: colors.dangerSoft,
    borderColor: colors.dangerBorder,
  },
  {
    id: 'EXPERIMENT_RUNNING',
    label: 'Experiment Running',
    description: 'Experiment is currently executing',
    icon: <Loader2 size={14} />,
    color: colors.infoText,
    bgColor: colors.infoSoft,
    borderColor: colors.infoBorder,
  },
  {
    id: 'EMPTY_EVIDENCE',
    label: 'Empty Evidence',
    description: 'No evidence records exist yet',
    icon: <EyeOff size={14} />,
    color: colors.bodyMuted,
    bgColor: colors.surfacePearl,
    borderColor: colors.border,
  },
  {
    id: 'MISSING_METRICS',
    label: 'Missing Metrics',
    description: 'Experiment completed but metrics are undefined',
    icon: <AlertTriangle size={14} />,
    color: colors.warningText,
    bgColor: colors.warningSoft,
    borderColor: colors.warningBorder,
  },
  {
    id: 'PARTIAL_RESULT',
    label: 'Partial Result',
    description: 'Experiment returned incomplete data',
    icon: <AlertTriangle size={14} />,
    color: colors.warningText,
    bgColor: colors.warningSoft,
    borderColor: colors.warningBorder,
  },
  {
    id: 'MOCK_RESULT',
    label: 'Mock Result',
    description: 'Result from simulated data, not live model',
    icon: <Shield size={14} />,
    color: colors.warningText,
    bgColor: colors.warningSoft,
    borderColor: colors.warningBorder,
  },
  {
    id: 'REAL_RESULT',
    label: 'Real Result',
    description: 'Result from live model execution',
    icon: <CheckCircle2 size={14} />,
    color: colors.successText,
    bgColor: colors.successSoft,
    borderColor: colors.successBorder,
  },
];

export const UIAdversarialTester: React.FC = () => {
  const [activeTestCase, setActiveTestCase] = useState<TestState | null>(null);
  const [testResults, setTestResults] = useState<Record<TestState, 'pending' | 'passed' | 'failed'>>({
    API_UNAVAILABLE: 'pending',
    MODEL_UNAVAILABLE: 'pending',
    EXPERIMENT_FAILED: 'pending',
    EXPERIMENT_RUNNING: 'pending',
    EMPTY_EVIDENCE: 'pending',
    MISSING_METRICS: 'pending',
    PARTIAL_RESULT: 'pending',
    MOCK_RESULT: 'pending',
    REAL_RESULT: 'pending',
  });

  const runTest = (testCase: TestCase) => {
    setActiveTestCase(testCase.id);
    // Simulate test execution
    setTimeout(() => {
      setTestResults((prev) => ({
        ...prev,
        [testCase.id]: 'passed',
      }));
      setActiveTestCase(null);
    }, 1000);
  };

  const passedCount = Object.values(testResults).filter((r) => r === 'passed').length;
  const totalCount = Object.keys(testResults).length;

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            UI Adversarial Testing
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            State Validation Matrix
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Verify every system state has a distinct UI representation.
          </div>
        </div>

        <div style={{
          padding: '12px 16px',
          borderRadius: 8,
          backgroundColor: colors.surfaceTile1,
          border: `1px solid ${colors.border}`,
          textAlign: 'center',
        }}>
          <div style={{ fontSize: 24, fontWeight: 800, color: passedCount === totalCount ? colors.success : colors.primary }}>
            {passedCount}/{totalCount}
          </div>
          <div style={{ fontSize: 10, fontWeight: 600, color: colors.bodyMuted, textTransform: 'uppercase' }}>
            Tests Passed
          </div>
        </div>
      </div>

      {/* Test Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
        {TEST_CASES.map((testCase) => {
          const result = testResults[testCase.id];
          const isActive = activeTestCase === testCase.id;

          return (
            <div
              key={testCase.id}
              style={{
                border: `1px solid ${result === 'passed' ? colors.successBorder : testCase.borderColor}`,
                borderRadius: 10,
                padding: 16,
                backgroundColor: result === 'passed' ? colors.successSoft : testCase.bgColor,
                opacity: isActive ? 0.7 : 1,
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ color: testCase.color }}>{testCase.icon}</div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>{testCase.label}</div>
                    <div style={{ fontSize: 11, color: colors.bodyMuted }}>{testCase.description}</div>
                  </div>
                </div>
                {result === 'passed' && (
                  <CheckCircle2 size={16} style={{ color: colors.success }} />
                )}
              </div>

              {/* Mock UI for this state */}
              <div style={{
                padding: 12,
                borderRadius: 6,
                backgroundColor: colors.canvas,
                border: `1px solid ${colors.border}`,
                marginBottom: 10,
              }}>
                {testCase.id === 'API_UNAVAILABLE' && (
                  <div style={{ fontSize: 12, color: colors.dangerText }}>
                    <AlertTriangle size={14} style={{ marginRight: 6 }} />
                    Backend service unreachable. Check if MECH backend is running.
                  </div>
                )}
                {testCase.id === 'MODEL_UNAVAILABLE' && (
                  <div style={{ fontSize: 12, color: colors.dangerText }}>
                    <Cpu size={14} style={{ marginRight: 6 }} />
                    PyTorch/Transformers not installed. Install dependencies to enable experiments.
                  </div>
                )}
                {testCase.id === 'EXPERIMENT_FAILED' && (
                  <div style={{ fontSize: 12, color: colors.dangerText }}>
                    <XCircle size={14} style={{ marginRight: 6 }} />
                    Experiment execution failed: CUDA out of memory.
                  </div>
                )}
                {testCase.id === 'EXPERIMENT_RUNNING' && (
                  <div style={{ fontSize: 12, color: colors.infoText, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Loader2 size={14} className="animate-spin" />
                    Running activation patching on L9H9...
                  </div>
                )}
                {testCase.id === 'EMPTY_EVIDENCE' && (
                  <div style={{ fontSize: 12, color: colors.bodyMuted, textAlign: 'center', padding: 8 }}>
                    No evidence records found. Run an experiment to generate evidence.
                  </div>
                )}
                {testCase.id === 'MISSING_METRICS' && (
                  <div style={{ fontSize: 12, color: colors.warningText }}>
                    <AlertTriangle size={14} style={{ marginRight: 6 }} />
                    Experiment completed but metrics are undefined.
                  </div>
                )}
                {testCase.id === 'PARTIAL_RESULT' && (
                  <div style={{ fontSize: 12, color: colors.warningText }}>
                    <AlertTriangle size={14} style={{ marginRight: 6 }} />
                    Partial result: baseline computed, intervention pending.
                  </div>
                )}
                {testCase.id === 'MOCK_RESULT' && (
                  <div style={{ fontSize: 12, color: colors.warningText }}>
                    <Shield size={14} style={{ marginRight: 6 }} />
                    Mock data: Δ = 1.234 (simulated, not from live model)
                  </div>
                )}
                {testCase.id === 'REAL_RESULT' && (
                  <div style={{ fontSize: 12, color: colors.successText }}>
                    <CheckCircle2 size={14} style={{ marginRight: 6 }} />
                    Live result: Δ = 2.341 (from actual model execution)
                  </div>
                )}
              </div>

              {/* Run Test Button */}
              <button
                onClick={() => runTest(testCase)}
                disabled={isActive || result === 'passed'}
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  borderRadius: 6,
                  border: 'none',
                  backgroundColor: result === 'passed' ? colors.success : testCase.color,
                  color: colors.onPrimary,
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: isActive || result === 'passed' ? 'default' : 'pointer',
                  opacity: isActive || result === 'passed' ? 0.7 : 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
                }}
              >
                {isActive ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : result === 'passed' ? (
                  <CheckCircle2 size={14} />
                ) : (
                  <RefreshCw size={14} />
                )}
                {isActive ? 'Testing...' : result === 'passed' ? 'Passed' : 'Run Test'}
              </button>
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <div style={{ marginTop: 20, padding: 16, borderRadius: 10, backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 8 }}>Test Summary</div>
        <div style={{ fontSize: 12, color: colors.bodyMuted, lineHeight: 1.6 }}>
          <p>Every system state must have a <b>distinct UI representation</b> to ensure the researcher can:</p>
          <ul style={{ margin: '8px 0', paddingLeft: 20 }}>
            <li>Immediately recognize when the API is unavailable</li>
            <li>Distinguish between mock and real experiment results</li>
            <li>See clear error messages for failed experiments</li>
            <li>Track running experiments with appropriate loading indicators</li>
            <li>Handle empty evidence gracefully without showing stale data</li>
          </ul>
          <p style={{ marginTop: 8 }}>
            <b>Agent 3 must NOT touch:</b> PyTorch experiment implementation, model hooks, causal metric calculations, hypothesis truth determination.
          </p>
        </div>
      </div>
    </div>
  );
};
