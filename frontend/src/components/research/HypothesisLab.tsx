import React, { useState } from 'react';
import { useResearchStore, Hypothesis } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import { Brain, Plus, Trash2, CheckCircle2, AlertTriangle, XCircle, ShieldAlert, Sparkles } from 'lucide-react';

export const HypothesisLab: React.FC = () => {
  const {
    activeInvestigation,
    hypotheses,
    activeHypothesisId,
    selectHypothesis,
    createHypothesis,
    deleteHypothesis,
    evaluateHypothesis,
  } = useResearchStore();

  const [title, setTitle] = useState('');
  const [component, setComponent] = useState('L9H9');
  const [statement, setStatement] = useState('');
  const [prediction, setPrediction] = useState('');
  const [expectedEvidence, setExpectedEvidence] = useState('');
  const [falsificationCondition, setFalsificationCondition] = useState('');
  const [evaluating, setEvaluating] = useState<string | null>(null);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !statement.trim()) return;
    await createHypothesis({
      title,
      target_component: component,
      statement,
      prediction,
      expected_evidence: expectedEvidence,
      falsification_condition: falsificationCondition,
    });
    setTitle('');
    setStatement('');
    setPrediction('');
    setExpectedEvidence('');
    setFalsificationCondition('');
  };

  const handleEvaluate = async (id: string) => {
    setEvaluating(id);
    try {
      await evaluateHypothesis(id);
    } finally {
      setEvaluating(null);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'SUPPORTED':
      case 'CAUSALLY_VERIFIED':
        return <span style={{ fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 999, backgroundColor: colors.successSoft, color: colors.successText, border: `1px solid ${colors.successBorder}` }}>SUPPORTED</span>;
      case 'PARTIALLY_SUPPORTED':
        return <span style={{ fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 999, backgroundColor: colors.warningSoft, color: colors.warningText, border: `1px solid ${colors.warningBorder}` }}>PARTIALLY SUPPORTED</span>;
      case 'CONTRADICTED':
      case 'FALSIFIED':
        return <span style={{ fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 999, backgroundColor: colors.dangerSoft, color: colors.dangerText, border: `1px solid ${colors.dangerBorder}` }}>CONTRADICTED</span>;
      default:
        return <span style={{ fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 999, backgroundColor: colors.surfacePearl, color: colors.bodyMuted, border: `1px solid ${colors.border}` }}>UNTESTED</span>;
    }
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
          Epistemic Lab
        </div>
        <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
          Hypothesis Formation & Falsification Engine
        </h2>
        <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
          Formulate falsifiable mechanistic claims, track predictions, and update epistemic beliefs based on live intervention evidence.
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 16 }}>
        {/* Create Hypothesis Form */}
        <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 0.4 }}>
            Formulate New Hypothesis
          </div>

          <form onSubmit={handleCreate} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Hypothesis Title
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. L9H9 Name Mover Role"
                style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Target Component Coordinate
              </label>
              <input
                type="text"
                value={component}
                onChange={(e) => setComponent(e.target.value)}
                placeholder="L9H9, L8_MLP, or Feature_412"
                style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Mechanistic Statement
              </label>
              <textarea
                value={statement}
                onChange={(e) => setStatement(e.target.value)}
                rows={2}
                placeholder="What role does this component play in the model's computation?"
                style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Specific Prediction
              </label>
              <textarea
                value={prediction}
                onChange={(e) => setPrediction(e.target.value)}
                rows={2}
                placeholder="What observable change in logits/probabilities should occur under intervention?"
                style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.dangerText, marginBottom: 4 }}>
                Falsification Condition (Crucial)
              </label>
              <textarea
                value={falsificationCondition}
                onChange={(e) => setFalsificationCondition(e.target.value)}
                rows={2}
                placeholder="What measurement or control result would disprove this hypothesis?"
                style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.dangerBorder}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
              />
            </div>

            <button
              type="submit"
              style={{
                marginTop: 6,
                padding: '8px 0',
                borderRadius: 6,
                border: 'none',
                backgroundColor: colors.primary,
                color: colors.onPrimary,
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
              }}
            >
              <Plus size={14} /> Register Hypothesis
            </button>
          </form>
        </div>

        {/* Existing Hypotheses List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4 }}>
            Registered Hypotheses ({hypotheses.length})
          </div>

          {hypotheses.map((h) => {
            const isSelected = h.id === activeHypothesisId;
            return (
              <div
                key={h.id}
                onClick={() => selectHypothesis(h.id)}
                style={{
                  border: `1px solid ${isSelected ? colors.primary : colors.border}`,
                  borderRadius: 10,
                  padding: 14,
                  backgroundColor: isSelected ? colors.surfacePearl : colors.surfaceTile1,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>{h.title}</span>
                    <code style={{ fontSize: 11, backgroundColor: colors.canvas, padding: '2px 6px', borderRadius: 4 }}>
                      {h.target_component}
                    </code>
                  </div>
                  {getStatusBadge(h.status)}
                </div>

                <div style={{ fontSize: 12, color: colors.body, marginBottom: 8, lineHeight: 1.4 }}>
                  {h.statement}
                </div>

                <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 8, backgroundColor: colors.canvas, padding: '6px 8px', borderRadius: 6 }}>
                  <b>Falsification criteria:</b> {h.falsification_condition || 'Not specified'}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, borderTop: `1px solid ${colors.borderLight}`, paddingTop: 8 }}>
                  <div style={{ display: 'flex', gap: 12 }}>
                    <span>Supporting: <b style={{ color: colors.successText }}>{h.evidence_count_supporting}</b></span>
                    <span>Contradicting: <b style={{ color: colors.dangerText }}>{h.evidence_count_contradicting}</b></span>
                  </div>

                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleEvaluate(h.id);
                      }}
                      disabled={evaluating === h.id}
                      style={{
                        padding: '4px 8px',
                        borderRadius: 4,
                        border: `1px solid ${colors.border}`,
                        backgroundColor: colors.canvas,
                        color: colors.ink,
                        fontSize: 10,
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      {evaluating === h.id ? 'Evaluating…' : 'Evaluate Evidence'}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteHypothesis(h.id);
                      }}
                      style={{
                        padding: '4px 6px',
                        borderRadius: 4,
                        border: 'none',
                        backgroundColor: 'transparent',
                        color: colors.dangerText,
                        cursor: 'pointer',
                      }}
                      title="Delete hypothesis"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
