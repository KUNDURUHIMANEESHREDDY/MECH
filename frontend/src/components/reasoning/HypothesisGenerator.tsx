import React, { FC, useState } from 'react';
import { Beaker, Brain, Lightbulb, ArrowRight, Plus, RefreshCw, AlertCircle, CheckCircle, Target } from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface Hypothesis {
  id: string;
  title: string;
  statement: string;
  target_component: string;
  prediction: string;
  falsification_condition: string;
  state: string;
  confidence: number;
  metadata: Record<string, unknown>;
}

export const HypothesisGenerator: FC = () => {
  const [behavior, setBehavior] = useState('');
  const [observation, setObservation] = useState('');
  const [researchQuestion, setResearchQuestion] = useState('');
  const [targetComponents, setTargetComponents] = useState('L9H9, L7H9');
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateHypotheses = async () => {
    if (!behavior || !observation || !researchQuestion) {
      setError('All fields are required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/reasoning/hypotheses/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          behavior,
          observation,
          research_question: researchQuestion,
          target_components: targetComponents.split(',').map(s => s.trim()),
          max_hypotheses: 3,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate hypotheses');
      }

      const data = await response.json();
      setHypotheses(data.hypotheses);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate hypotheses');
    } finally {
      setLoading(false);
    }
  };

  const stateColors: Record<string, string> = {
    PROPOSED: colors.purple,
    PLANNED: colors.primary,
    TESTING: colors.warning,
    REPLICATION: colors.pink,
    EVALUATION: colors.infoSoft,
    SUPPORTED: colors.success,
    REFUTED: colors.danger,
    INSUFFICIENT: colors.inkMuted80,
  };

  return (
    <div style={{ padding: '16px', height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <Brain size={20} style={{ color: colors.purple }} />
        <h2 style={{ fontSize: '16px', fontWeight: 600, color: colors.ink, margin: 0 }}>
          Hypothesis Generator
        </h2>
      </div>

      <div style={{
        background: colors.surfacePearl,
        border: `1px solid ${colors.border}`,
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '16px',
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div>
            <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
              Behavior Being Investigated
            </label>
            <input
              value={behavior}
              onChange={(e) => setBehavior(e.target.value)}
              placeholder="e.g., Indirect Object Identification"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: colors.canvas,
                border: `1px solid ${colors.border}`,
                borderRadius: '6px',
                color: colors.ink,
                fontSize: '13px',
              }}
            />
          </div>

          <div>
            <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
              What Was Observed
            </label>
            <input
              value={observation}
              onChange={(e) => setObservation(e.target.value)}
              placeholder="e.g., L9H9 has high attribution to indirect object token"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: colors.canvas,
                border: `1px solid ${colors.border}`,
                borderRadius: '6px',
                color: colors.ink,
                fontSize: '13px',
              }}
            />
          </div>

          <div>
            <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
              Research Question
            </label>
            <input
              value={researchQuestion}
              onChange={(e) => setResearchQuestion(e.target.value)}
              placeholder="e.g., Does L9H9 causally contribute to IOI?"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: colors.canvas,
                border: `1px solid ${colors.border}`,
                borderRadius: '6px',
                color: colors.ink,
                fontSize: '13px',
              }}
            />
          </div>

          <div>
            <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
              Target Components (comma-separated)
            </label>
            <input
              value={targetComponents}
              onChange={(e) => setTargetComponents(e.target.value)}
              placeholder="L9H9, L7H9, L8H1"
              style={{
                width: '100%',
                padding: '8px 12px',
                background: colors.canvas,
                border: `1px solid ${colors.border}`,
                borderRadius: '6px',
                color: colors.ink,
                fontSize: '13px',
              }}
            />
          </div>

          <button
            onClick={generateHypotheses}
            disabled={loading}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '10px 16px',
              background: colors.purple,
              color: colors.onPrimary,
              border: 'none',
              borderRadius: '6px',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.6 : 1,
              fontSize: '13px',
              fontWeight: 500,
            }}
          >
            {loading ? <RefreshCw size={16} className="animate-spin" /> : <Lightbulb size={16} />}
            Generate Hypotheses
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '12px',
          background: colors.dangerSoft,
          border: `1px solid ${colors.dangerBorder}`,
          borderRadius: '6px',
          marginBottom: '16px',
        }}>
          <AlertCircle size={16} style={{ color: colors.danger }} />
          <span style={{ color: colors.dangerText, fontSize: '13px' }}>{error}</span>
        </div>
      )}

      {hypotheses.length > 0 && (
        <div>
          <h3 style={{ fontSize: '14px', color: colors.bodyMuted, marginBottom: '12px' }}>
            Generated Hypotheses ({hypotheses.length})
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {hypotheses.map((hyp) => (
              <div
                key={hyp.id}
                style={{
                  background: colors.surfacePearl,
                  border: `1px solid ${colors.border}`,
                  borderRadius: '8px',
                  padding: '16px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                  <h4 style={{ fontSize: '14px', color: colors.ink, margin: 0 }}>{hyp.title}</h4>
                  <span style={{
                    fontSize: '11px',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    background: `${stateColors[hyp.state] || colors.inkMuted80}20`,
                    color: stateColors[hyp.state] || colors.inkMuted80,
                  }}>
                    {hyp.state}
                  </span>
                </div>

                <p style={{ fontSize: '13px', color: colors.bodyMuted, margin: '0 0 12px 0', lineHeight: '1.5' }}>
                  {hyp.statement}
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div style={{
                    background: colors.canvas,
                    padding: '10px',
                    borderRadius: '6px',
                    border: `1px solid ${colors.hairline}`,
                  }}>
                    <div style={{ fontSize: '11px', color: colors.inkMuted80, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Target size={12} /> Prediction
                    </div>
                    <p style={{ fontSize: '12px', color: colors.bodyMuted, margin: 0, lineHeight: '1.4' }}>
                      {hyp.prediction}
                    </p>
                  </div>

                  <div style={{
                    background: colors.canvas,
                    padding: '10px',
                    borderRadius: '6px',
                    border: `1px solid ${colors.hairline}`,
                  }}>
                    <div style={{ fontSize: '11px', color: colors.inkMuted80, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <AlertCircle size={12} /> Falsification Condition
                    </div>
                    <p style={{ fontSize: '12px', color: colors.bodyMuted, margin: 0, lineHeight: '1.4' }}>
                      {hyp.falsification_condition}
                    </p>
                  </div>
                </div>

                <div style={{
                  marginTop: '12px',
                  padding: '8px 12px',
                  background: colors.canvas,
                  borderRadius: '4px',
                  border: `1px solid ${colors.hairline}`,
                }}>
                  <span style={{ fontSize: '11px', color: colors.inkMuted80 }}>
                    Target: <span style={{ color: colors.primary }}>{hyp.target_component}</span>
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
