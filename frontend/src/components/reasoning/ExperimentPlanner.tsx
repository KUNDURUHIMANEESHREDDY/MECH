import React, { FC, useState, useEffect } from 'react';
import { ClipboardList, Play, RefreshCw, AlertCircle, CheckCircle, ArrowRight, Layers, Repeat } from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface ExperimentSpec {
  id: string;
  hypothesis_id: string;
  name: string;
  description: string;
  clean_prompt: string;
  corrupted_prompt: string | null;
  target_token: string;
  distractor_token: string | null;
  source_component: string;
  intervention_type: string;
  control_components: string[];
  repeats: number;
  seed: number;
  metrics: string[];
}

interface Hypothesis {
  id: string;
  title: string;
  statement: string;
  target_component: string;
  state: string;
}

export const ExperimentPlanner: FC = () => {
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [selectedHypothesis, setSelectedHypothesis] = useState<Hypothesis | null>(null);
  const [cleanPrompt, setCleanPrompt] = useState('When Mary saw Tom, she told him that');
  const [targetToken, setTargetToken] = useState('him');
  const [repeats, setRepeats] = useState(10);
  const [experimentSpec, setExperimentSpec] = useState<ExperimentSpec | null>(null);
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

  const planExperiment = async () => {
    if (!selectedHypothesis) {
      setError('Select a hypothesis first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/reasoning/experiments/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hypothesis_id: selectedHypothesis.id,
          clean_prompt: cleanPrompt,
          target_token: targetToken,
          repeats,
          seed: 42,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to plan experiment');
      }

      const data = await response.json();
      setExperimentSpec(data.experiment_spec);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to plan experiment');
    } finally {
      setLoading(false);
    }
  };

  const executeExperiment = async () => {
    if (!experimentSpec) return;

    setExecuting(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/research/experiments/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hypothesis_id: experimentSpec.hypothesis_id,
          clean_prompt: experimentSpec.clean_prompt,
          corrupted_prompt: experimentSpec.corrupted_prompt,
          target_token: experimentSpec.target_token,
          distractor_token: experimentSpec.distractor_token,
          source_component: experimentSpec.source_component,
          intervention_type: experimentSpec.intervention_type,
          control_components: experimentSpec.control_components,
          repeats: experimentSpec.repeats,
          seed: experimentSpec.seed,
          metrics: experimentSpec.metrics,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to execute experiment');
      }

      const data = await response.json();
      alert(`Experiment completed! Status: ${data.run?.execution_status || 'UNKNOWN'}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to execute experiment');
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div style={{ padding: '16px', height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <ClipboardList size={20} style={{ color: colors.primary }} />
        <h2 style={{ fontSize: '16px', fontWeight: 600, color: colors.ink, margin: 0 }}>
          Experiment Planner
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
                background: selectedHypothesis?.id === hyp.id ? `${colors.primary}20` : colors.canvas,
                border: `1px solid ${selectedHypothesis?.id === hyp.id ? colors.primary : colors.hairline}`,
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
          2. Configure Experiment
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

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
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
            <div>
              <label style={{ fontSize: '12px', color: colors.bodyMuted, marginBottom: '4px', display: 'block' }}>
                Repeats
              </label>
              <input
                type="number"
                value={repeats}
                onChange={(e) => setRepeats(parseInt(e.target.value) || 10)}
                min={1}
                max={100}
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
          </div>

          <button
            onClick={planExperiment}
            disabled={loading || !selectedHypothesis}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px',
              padding: '10px 16px',
              background: colors.primary,
              color: colors.onPrimary,
              border: 'none',
              borderRadius: '6px',
              cursor: loading || !selectedHypothesis ? 'not-allowed' : 'pointer',
              opacity: loading || !selectedHypothesis ? 0.6 : 1,
              fontSize: '13px',
              fontWeight: 500,
            }}
          >
            {loading ? <RefreshCw size={16} className="animate-spin" /> : <ClipboardList size={16} />}
            Plan Experiment
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

      {experimentSpec && (
        <div style={{
          background: colors.surfacePearl,
          border: `1px solid ${colors.border}`,
          borderRadius: '8px',
          padding: '16px',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h3 style={{ fontSize: '14px', color: colors.bodyMuted, margin: 0 }}>
              3. Experiment Specification
            </h3>
            <button
              onClick={executeExperiment}
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
              {executing ? <RefreshCw size={14} className="animate-spin" /> : <Play size={14} />}
              Execute on Agent 1
            </button>
          </div>

          <div style={{
            background: colors.canvas,
            padding: '12px',
            borderRadius: '6px',
            border: `1px solid ${colors.hairline}`,
            marginBottom: '12px',
          }}>
            <div style={{ fontSize: '13px', color: colors.ink, fontWeight: 500, marginBottom: '8px' }}>
              {experimentSpec.name}
            </div>
            <div style={{ fontSize: '12px', color: colors.inkMuted80, lineHeight: '1.5' }}>
              {experimentSpec.description}
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
                <Layers size={12} /> Intervention
              </div>
              <div style={{ fontSize: '12px', color: colors.bodyMuted }}>
                {experimentSpec.source_component} → {experimentSpec.intervention_type}
              </div>
            </div>

            <div style={{
              background: colors.canvas,
              padding: '10px',
              borderRadius: '6px',
              border: `1px solid ${colors.hairline}`,
            }}>
              <div style={{ fontSize: '11px', color: colors.inkMuted80, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Repeat size={12} /> Controls
              </div>
              <div style={{ fontSize: '12px', color: colors.bodyMuted }}>
                {experimentSpec.control_components.join(', ')}
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
              Metrics: {experimentSpec.metrics.join(', ')}
            </div>
            <div style={{ fontSize: '11px', color: colors.inkMuted80 }}>
              Repeats: {experimentSpec.repeats} | Seed: {experimentSpec.seed}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
