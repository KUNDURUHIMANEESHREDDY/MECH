import React, { FC, useState, useEffect } from 'react';
import { ShieldAlert, RefreshCw, AlertCircle, CheckCircle, ArrowRight, Target, Zap } from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface Hypothesis {
  id: string;
  title: string;
  statement: string;
  target_component: string;
  prediction: string;
  falsification_condition: string;
  state: string;
}

interface FalsificationSpec {
  id: string;
  hypothesis_id: string;
  name: string;
  description: string;
  clean_prompt: string;
  source_component: string;
  control_components: string[];
  repeats: number;
  metrics: string[];
}

export const FalsificationDesigner: FC = () => {
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [selectedHypothesis, setSelectedHypothesis] = useState<Hypothesis | null>(null);
  const [cleanPrompt, setCleanPrompt] = useState('When Mary saw Tom, she told him that');
  const [targetToken, setTargetToken] = useState('him');
  const [falsificationSpec, setFalsificationSpec] = useState<FalsificationSpec | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [executing, setExecuting] = useState(false);

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

  const designFalsification = async () => {
    if (!selectedHypothesis) {
      setError('Select a hypothesis first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/reasoning/hypotheses/falsification', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hypothesis_id: selectedHypothesis.id,
          clean_prompt: cleanPrompt,
          target_token: targetToken,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to design falsification test');
      }

      const data = await response.json();
      setFalsificationSpec(data.falsification_spec);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to design falsification test');
    } finally {
      setLoading(false);
    }
  };

  const executeFalsification = async () => {
    if (!falsificationSpec) return;

    setExecuting(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/research/experiments/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hypothesis_id: falsificationSpec.hypothesis_id,
          clean_prompt: falsificationSpec.clean_prompt,
          target_token: targetToken,
          source_component: falsificationSpec.source_component,
          intervention_type: 'ABLATION_ZERO',
          control_components: falsificationSpec.control_components,
          repeats: falsificationSpec.repeats,
          seed: 42,
          metrics: falsificationSpec.metrics,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to execute falsification test');
      }

      const data = await response.json();
      alert(`Falsification test completed! Status: ${data.run?.execution_status || 'UNKNOWN'}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to execute falsification test');
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div style={{ padding: '16px', height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <ShieldAlert size={20} style={{ color: colors.danger }} />
        <h2 style={{ fontSize: '16px', fontWeight: 600, color: colors.ink, margin: 0 }}>
          Falsification Designer
        </h2>
      </div>

      <div style={{
        background: `${colors.danger}10`,
        border: `1px solid ${colors.danger}30`,
        borderRadius: '8px',
        padding: '12px',
        marginBottom: '16px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <ShieldAlert size={16} style={{ color: colors.danger }} />
          <span style={{ fontSize: '13px', color: colors.danger, fontWeight: 500 }}>Falsification Philosophy</span>
        </div>
        <p style={{ fontSize: '12px', color: colors.bodyMuted, margin: 0, lineHeight: '1.5' }}>
          Instead of asking "How can I prove this hypothesis?", we ask "What experiment could disprove this?"
          A strong hypothesis makes specific predictions that can be tested and potentially falsified.
        </p>
      </div>

      <div style={{
        background: colors.surfacePearl,
        border: `1px solid ${colors.border}`,
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '16px',
      }}>
        <h3 style={{ fontSize: '14px', color: colors.bodyMuted, marginBottom: '12px' }}>
          1. Select Hypothesis to Falsify
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '200px', overflow: 'auto' }}>
          {hypotheses.map((hyp) => (
            <div
              key={hyp.id}
              onClick={() => setSelectedHypothesis(hyp)}
              style={{
                padding: '10px 12px',
                background: selectedHypothesis?.id === hyp.id ? `${colors.danger}20` : colors.canvas,
                border: `1px solid ${selectedHypothesis?.id === hyp.id ? colors.danger : colors.hairline}`,
                borderRadius: '6px',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <div style={{ fontSize: '13px', color: colors.ink, fontWeight: 500 }}>{hyp.title}</div>
              <div style={{ fontSize: '12px', color: colors.inkMuted80, marginTop: '4px' }}>
                Falsification: {hyp.falsification_condition}
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
          2. Configure Falsification Test
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div>
            <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
              Clean Prompt
            </label>
            <input
              value={cleanPrompt}
              onChange={(e) => setCleanPrompt(e.target.value)}
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
              Target Token
            </label>
            <input
              value={targetToken}
              onChange={(e) => setTargetToken(e.target.value)}
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
            onClick={designFalsification}
            disabled={loading || !selectedHypothesis}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '10px 16px',
              background: colors.danger,
              color: colors.onPrimary,
              border: 'none',
              borderRadius: '6px',
              cursor: loading || !selectedHypothesis ? 'not-allowed' : 'pointer',
              opacity: loading || !selectedHypothesis ? 0.6 : 1,
              fontSize: '13px',
              fontWeight: 500,
            }}
          >
            {loading ? <RefreshCw size={16} className="animate-spin" /> : <ShieldAlert size={16} />}
            Design Falsification Test
          </button>
        </div>
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

      {falsificationSpec && (
        <div style={{
          background: colors.surfacePearl,
          border: `1px solid ${colors.border}`,
          borderRadius: '8px',
          padding: '16px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ fontSize: '14px', color: colors.bodyMuted, margin: 0 }}>
              3. Falsification Specification
            </h3>
            <button
              onClick={executeFalsification}
              disabled={executing}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px 12px',
                background: colors.success,
                color: colors.onPrimary,
                border: 'none',
                borderRadius: '6px',
                cursor: executing ? 'not-allowed' : 'pointer',
                opacity: executing ? 0.6 : 1,
                fontSize: '12px',
                fontWeight: 500,
              }}
            >
              {executing ? <RefreshCw size={14} className="animate-spin" /> : <Zap size={14} />}
              Execute on Agent 1
            </button>
          </div>

          <div style={{
            background: `${colors.danger}10`,
            padding: '12px',
            borderRadius: '6px',
            border: `1px solid ${colors.danger}30`,
            marginBottom: '12px',
          }}>
            <div style={{ fontSize: '13px', color: colors.ink, fontWeight: 500, marginBottom: '8px' }}>
              {falsificationSpec.name}
            </div>
            <div style={{ fontSize: '12px', color: colors.bodyMuted, lineHeight: '1.5' }}>
              {falsificationSpec.description}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
            <div style={{
              background: colors.canvas,
              padding: '10px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
            }}>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Target size={12} /> Intervention
              </div>
              <div style={{ fontSize: '12px', color: colors.bodyMuted }}>
                {falsificationSpec.source_component} → ABLATION_ZERO
              </div>
            </div>

            <div style={{
              background: colors.canvas,
              padding: '10px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
            }}>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Target size={12} /> Controls
              </div>
              <div style={{ fontSize: '12px', color: colors.bodyMuted }}>
                {falsificationSpec.control_components.join(', ')}
              </div>
            </div>
          </div>

          <div style={{
            background: colors.canvas,
            padding: '10px',
            borderRadius: '6px',
            border: `1px solid ${colors.hairline}`,
          }}>
            <div style={{ fontSize: '11px', color: colors.inkMuted80, marginBottom: '4px' }}>
              Metrics: {falsificationSpec.metrics.join(', ')}
            </div>
            <div style={{ fontSize: '11px', color: colors.inkMuted80 }}>
              Repeats: {falsificationSpec.repeats} (higher for falsification)
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
