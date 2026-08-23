import React, { FC, useState, useEffect } from 'react';
import { Network, RefreshCw, AlertCircle, CheckCircle, XCircle, MinusCircle, ArrowRight, Layers } from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface Hypothesis {
  id: string;
  title: string;
  statement: string;
  target_component: string;
  state: string;
}

interface EvidenceChain {
  hypothesis_id: string;
  supporting_count: number;
  contradicting_count: number;
  total_effect_size: number;
  specificity_ratio: number;
  replication_count: number;
  overall_assessment: string;
}

export const EvidenceReasoner: FC = () => {
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [selectedHypothesis, setSelectedHypothesis] = useState<Hypothesis | null>(null);
  const [evidenceChain, setEvidenceChain] = useState<EvidenceChain | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHypotheses();
  }, []);

  const fetchHypotheses = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/reasoning/hypotheses');
      if (response.ok) {
        const data = await response.json();
        setHypotheses(data.hypotheses);
      }
    } catch (err) {
      console.error('Failed to fetch hypotheses:', err);
    }
  };

  const reasonAboutEvidence = async () => {
    if (!selectedHypothesis) {
      setError('Select a hypothesis first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/reasoning/evidence/reason', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hypothesis_id: selectedHypothesis.id,
          evidence_records: [],
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to reason about evidence');
      }

      const data = await response.json();
      setEvidenceChain(data.evidence_chain);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reason about evidence');
    } finally {
      setLoading(false);
    }
  };

  const assessmentColors: Record<string, string> = {
    EVIDENCE_SUPPORTS: colors.success,
    EVIDENCE_CONTRADICTS: colors.danger,
    INSUFFICIENT_EVIDENCE: colors.warning,
  };

  const assessmentIcons: Record<string, React.ReactNode> = {
    EVIDENCE_SUPPORTS: <CheckCircle size={16} />,
    EVIDENCE_CONTRADICTS: <XCircle size={16} />,
    INSUFFICIENT_EVIDENCE: <MinusCircle size={16} />,
  };

  return (
    <div style={{ padding: '16px', height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <Network size={20} style={{ color: colors.infoSoft }} />
        <h2 style={{ fontSize: '16px', fontWeight: 600, color: colors.ink, margin: 0 }}>
          Evidence Reasoner
        </h2>
      </div>

      <div style={{
        background: colors.surfacePearl,
        border: `1px solid ${colors.border}`,
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '16px',
      }}>
        <h3 style={{ fontSize: '14px', color: colors.bodyMuted, marginBottom: '12px' }}>
          1. Select Hypothesis
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '200px', overflow: 'auto' }}>
          {hypotheses.map((hyp) => (
            <div
              key={hyp.id}
              onClick={() => setSelectedHypothesis(hyp)}
              style={{
                padding: '10px 12px',
                background: selectedHypothesis?.id === hyp.id ? `${colors.infoSoft}20` : colors.canvas,
                border: `1px solid ${selectedHypothesis?.id === hyp.id ? colors.infoSoft : colors.hairline}`,
                borderRadius: '6px',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ fontSize: '13px', color: colors.ink, fontWeight: 500 }}>{hyp.title}</div>
              <div style={{ fontSize: '12px', color: colors.inkMuted80, marginTop: '4px' }}>
                Target: {hyp.target_component} | State: {hyp.state}
              </div>
            </div>
          ))}
          {hypotheses.length === 0 && (
            <div style={{ fontSize: '13px', color: colors.inkMuted80, textAlign: 'center', padding: '20px' }}>
              No hypotheses yet. Generate some in the Hypothesis Generator first.
            </div>
          )}
        </div>
      </div>

      <div style={{
        background: colors.surfacePearl,
        border: `1px solid ${colors.border}`,
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '16px',
      }}>
        <h3 style={{ fontSize: '14px', color: colors.bodyMuted, marginBottom: '12px' }}>
          2. Analyze Evidence Chain
        </h3>
        <button
          onClick={reasonAboutEvidence}
          disabled={loading || !selectedHypothesis}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px',
            padding: '10px 16px',
            background: colors.infoSoft,
            color: colors.onPrimary,
            border: 'none',
            borderRadius: '6px',
            cursor: loading || !selectedHypothesis ? 'not-allowed' : 'pointer',
            opacity: loading || !selectedHypothesis ? 0.6 : 1,
            fontSize: '13px',
            fontWeight: 500,
          }}
        >
          {loading ? <RefreshCw size={16} className="animate-spin" /> : <Network size={16} />}
          Reason About Evidence
        </button>
      </div>

      {error && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '12px',
          background: `${colors.danger}15`,
          border: `1px solid ${colors.danger}40`,
          borderRadius: '6px',
          marginBottom: '16px',
        }}>
          <AlertCircle size={16} style={{ color: colors.danger }} />
          <span style={{ color: colors.danger, fontSize: '13px' }}>{error}</span>
        </div>
      )}

      {evidenceChain && (
        <div style={{
          background: colors.surfacePearl,
          border: `1px solid ${colors.border}`,
          borderRadius: '8px',
          padding: '16px',
        }}>
          <h3 style={{ fontSize: '14px', color: colors.bodyMuted, marginBottom: '12px' }}>
            3. Evidence Chain Analysis
          </h3>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '12px',
            background: `${assessmentColors[evidenceChain.overall_assessment] || colors.inkMuted80}15`,
            border: `1px solid ${assessmentColors[evidenceChain.overall_assessment] || colors.inkMuted80}40`,
            borderRadius: '6px',
            marginBottom: '16px',
          }}>
            <span style={{ color: assessmentColors[evidenceChain.overall_assessment] || colors.inkMuted80 }}>
              {assessmentIcons[evidenceChain.overall_assessment]}
            </span>
            <span style={{
              fontSize: '14px',
              fontWeight: 600,
              color: assessmentColors[evidenceChain.overall_assessment] || colors.inkMuted80,
            }}>
              {evidenceChain.overall_assessment.replace(/_/g, ' ')}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '16px' }}>
            <div style={{
              background: colors.canvas,
              padding: '12px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '24px', fontWeight: 700, color: colors.success }}>
                {evidenceChain.supporting_count}
              </div>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginTop: '4px' }}>Supporting</div>
            </div>

            <div style={{
              background: colors.canvas,
              padding: '12px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '24px', fontWeight: 700, color: colors.danger }}>
                {evidenceChain.contradicting_count}
              </div>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginTop: '4px' }}>Contradicting</div>
            </div>

            <div style={{
              background: colors.canvas,
              padding: '12px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '24px', fontWeight: 700, color: colors.primary }}>
                {evidenceChain.replication_count}
              </div>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginTop: '4px' }}>Replications</div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div style={{
              background: colors.canvas,
              padding: '10px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
            }}>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginBottom: '4px' }}>
                Total Effect Size
              </div>
              <div style={{ fontSize: '14px', color: colors.ink, fontWeight: 600 }}>
                {evidenceChain.total_effect_size.toFixed(3)}
              </div>
            </div>

            <div style={{
              background: colors.canvas,
              padding: '10px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
            }}>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginBottom: '4px' }}>
                Specificity Ratio
              </div>
              <div style={{
                fontSize: '14px',
                color: evidenceChain.specificity_ratio > 2 ? colors.success : colors.warning,
                fontWeight: 600,
              }}>
                {evidenceChain.specificity_ratio.toFixed(2)}x
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
