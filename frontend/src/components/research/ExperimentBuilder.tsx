import React, { useState, useCallback } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import {
  FlaskConical,
  Play,
  Settings2,
  Database,
  Brain,
  Target,
  Sliders,
  BarChart3,
  AlertCircle,
  CheckCircle2,
  XCircle,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Layers,
  Shield,
  Hash,
} from 'lucide-react';

type Tab = 'model' | 'dataset' | 'hypothesis' | 'intervention' | 'controls' | 'metrics';

interface ExperimentConfig {
  name: string;
  description: string;
  model_id: string;
  dataset_id: string;
  clean_prompt: string;
  corrupted_prompt: string;
  target_token: string;
  distractor_token: string;
  source_component: string;
  intervention_type: string;
  steering_coefficient: number;
  control_component: string;
  control_components: string[];
  repeats: number;
  seed: number;
  selected_metrics: string[];
  hypothesis_id: string;
}

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'model', label: 'Model', icon: <Layers size={14} /> },
  { id: 'dataset', label: 'Dataset', icon: <Database size={14} /> },
  { id: 'hypothesis', label: 'Hypothesis', icon: <Brain size={14} /> },
  { id: 'intervention', label: 'Intervention', icon: <Target size={14} /> },
  { id: 'controls', label: 'Controls', icon: <Shield size={14} /> },
  { id: 'metrics', label: 'Metrics', icon: <BarChart3 size={14} /> },
];

const INTERVENTION_TYPES = [
  { value: 'ACTIVATION_PATCHING', label: 'Activation Patching', description: 'Replace clean activation with corrupted' },
  { value: 'ABLATION_ZERO', label: 'Zero Ablation', description: 'Set component output to zero' },
  { value: 'ABLATION_MEAN', label: 'Mean Ablation', description: 'Replace with dataset mean' },
  { value: 'STEERING', label: 'Activation Steering', description: 'Scale activation by coefficient' },
];

const AVAILABLE_METRICS = [
  { id: 'delta_logit', label: 'Δ Logit', description: 'Change in logit difference between target and distractor' },
  { id: 'delta_prob', label: 'Δ Probability', description: 'Change in target token probability' },
  { id: 'effect_size', label: 'Effect Size (Cohen\'s d)', description: 'Standardized effect magnitude' },
  { id: 'specificity', label: 'Specificity Ratio', description: 'Target effect / max control effect' },
  { id: 'cross_prompt_stability', label: 'Cross-Prompt Stability', description: 'Effect consistency across prompts' },
  { id: 'p_value', label: 'p-value', description: 'Statistical significance' },
];

const DEFAULT_CONTROL_COMPONENTS = ['L0H0', 'L6H6', 'L11H11'];

export const ExperimentBuilder: React.FC = () => {
  const {
    activeInvestigation,
    activeHypothesis,
    hypotheses,
    loading,
    error: storeError,
    runCausalExperiment,
  } = useResearchStore();

  const [activeTab, setActiveTab] = useState<Tab>('model');
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);

  const [config, setConfig] = useState<ExperimentConfig>({
    name: '',
    description: '',
    model_id: 'gpt2',
    dataset_id: activeInvestigation?.dataset_id || 'ioi',
    clean_prompt: 'When Mary and John went to the store, John gave a drink to',
    corrupted_prompt: 'When Mary and John went to the store, Mary gave a drink to',
    target_token: ' Mary',
    distractor_token: ' John',
    source_component: 'L9H9',
    intervention_type: 'ACTIVATION_PATCHING',
    steering_coefficient: 1.0,
    control_component: 'L9H8',
    control_components: [...DEFAULT_CONTROL_COMPONENTS],
    repeats: 3,
    seed: 42,
    selected_metrics: ['delta_logit', 'delta_prob', 'effect_size'],
    hypothesis_id: activeHypothesis?.id || '',
  });

  const updateConfig = useCallback((field: keyof ExperimentConfig, value: any) => {
    setConfig((prev) => ({ ...prev, [field]: value }));
  }, []);

  const addControlComponent = () => {
    updateConfig('control_components', [...config.control_components, '']);
  };

  const removeControlComponent = (idx: number) => {
    updateConfig(
      'control_components',
      config.control_components.filter((_, i) => i !== idx)
    );
  };

  const updateControlComponent = (idx: number, value: string) => {
    const updated = [...config.control_components];
    updated[idx] = value;
    updateConfig('control_components', updated);
  };

  const toggleMetric = (metricId: string) => {
    setConfig((prev) => ({
      ...prev,
      selected_metrics: prev.selected_metrics.includes(metricId)
        ? prev.selected_metrics.filter((m) => m !== metricId)
        : [...prev.selected_metrics, metricId],
    }));
  };

  const handleRun = async () => {
    if (!config.name.trim()) {
      setError('Experiment name is required.');
      return;
    }
    if (!config.clean_prompt.trim()) {
      setError('Clean prompt is required.');
      return;
    }
    if (!config.target_token.trim()) {
      setError('Target token is required.');
      return;
    }

    setRunning(true);
    setError(null);
    setResult(null);

    try {
      const run = await runCausalExperiment({
        name: config.name,
        description: config.description,
        model_id: config.model_id,
        dataset_id: config.dataset_id,
        clean_prompt: config.clean_prompt,
        corrupted_prompt: config.corrupted_prompt || undefined,
        target_token: config.target_token,
        distractor_token: config.distractor_token || undefined,
        source_component: config.source_component,
        intervention_type: config.intervention_type,
        steering_coefficient: config.steering_coefficient,
        control_component: config.control_component,
        control_components: config.control_components.filter(Boolean),
        repeats: config.repeats,
        seed: config.seed,
        selected_metrics: config.selected_metrics,
        hypothesis_id: config.hypothesis_id || undefined,
      });
      setResult(run);
    } catch (err: any) {
      setError(err.message || 'Experiment execution failed.');
    } finally {
      setRunning(false);
    }
  };

  const handleReset = () => {
    setConfig({
      name: '',
      description: '',
      model_id: 'gpt2',
      dataset_id: activeInvestigation?.dataset_id || 'ioi',
      clean_prompt: '',
      corrupted_prompt: '',
      target_token: '',
      distractor_token: '',
      source_component: 'L9H9',
      intervention_type: 'ACTIVATION_PATCHING',
      steering_coefficient: 1.0,
      control_component: 'L9H8',
      control_components: [...DEFAULT_CONTROL_COMPONENTS],
      repeats: 3,
      seed: 42,
      selected_metrics: ['delta_logit', 'delta_prob', 'effect_size'],
      hypothesis_id: activeHypothesis?.id || '',
    });
    setError(null);
    setResult(null);
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'model':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <FieldGroup label="Model ID">
              <select
                value={config.model_id}
                onChange={(e) => updateConfig('model_id', e.target.value)}
                style={selectStyle}
              >
                <option value="gpt2">GPT-2 Small (124M)</option>
                <option value="gpt2-medium">GPT-2 Medium (355M)</option>
                <option value="gpt2-large">GPT-2 Large (774M)</option>
                <option value="gpt2-xl">GPT-2 XL (1.5B)</option>
              </select>
            </FieldGroup>

            <FieldGroup label="Experiment Name" required>
              <input
                type="text"
                value={config.name}
                onChange={(e) => updateConfig('name', e.target.value)}
                placeholder="e.g., IOI Name Mover Ablation Study"
                style={inputStyle}
              />
            </FieldGroup>

            <FieldGroup label="Description">
              <textarea
                value={config.description}
                onChange={(e) => updateConfig('description', e.target.value)}
                rows={3}
                placeholder="Brief description of the experiment's purpose..."
                style={{ ...inputStyle, resize: 'vertical' }}
              />
            </FieldGroup>

            <FieldGroup label="Random Seed">
              <input
                type="number"
                value={config.seed}
                onChange={(e) => updateConfig('seed', parseInt(e.target.value, 10) || 42)}
                style={inputStyle}
              />
            </FieldGroup>
          </div>
        );

      case 'dataset':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <FieldGroup label="Dataset">
              <select
                value={config.dataset_id}
                onChange={(e) => updateConfig('dataset_id', e.target.value)}
                style={selectStyle}
              >
                <option value="ioi">Indirect Object Identification (IOI)</option>
                <option value="induction">Induction (Copying)</option>
                <option value="factual">Factual Recall</option>
                <option value="semantic">Semantic Relations</option>
                <option value="syntax">Syntax / Agreement</option>
                <option value="custom">Custom Dataset</option>
              </select>
            </FieldGroup>

            <FieldGroup label="Clean Prompt" required>
              <textarea
                value={config.clean_prompt}
                onChange={(e) => updateConfig('clean_prompt', e.target.value)}
                rows={3}
                placeholder="The prompt that produces the correct behavior..."
                style={{ ...inputStyle, resize: 'vertical' }}
              />
            </FieldGroup>

            <FieldGroup label="Corrupted Prompt">
              <textarea
                value={config.corrupted_prompt}
                onChange={(e) => updateConfig('corrupted_prompt', e.target.value)}
                rows={3}
                placeholder="The prompt with disrupted behavior (for patching experiments)..."
                style={{ ...inputStyle, resize: 'vertical' }}
              />
            </FieldGroup>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <FieldGroup label="Target Token" required>
                <input
                  type="text"
                  value={config.target_token}
                  onChange={(e) => updateConfig('target_token', e.target.value)}
                  placeholder=" Mary"
                  style={inputStyle}
                />
              </FieldGroup>
              <FieldGroup label="Distractor Token">
                <input
                  type="text"
                  value={config.distractor_token}
                  onChange={(e) => updateConfig('distractor_token', e.target.value)}
                  placeholder=" John"
                  style={inputStyle}
                />
              </FieldGroup>
            </div>
          </div>
        );

      case 'hypothesis':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <FieldGroup label="Link to Hypothesis">
              <select
                value={config.hypothesis_id}
                onChange={(e) => updateConfig('hypothesis_id', e.target.value)}
                style={selectStyle}
              >
                <option value="">No hypothesis linked</option>
                {hypotheses.map((h) => (
                  <option key={h.id} value={h.id}>
                    {h.title} ({h.target_component})
                  </option>
                ))}
              </select>
            </FieldGroup>

            {config.hypothesis_id && (
              <div style={cardStyle}>
                <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.4, marginBottom: 6 }}>
                  Linked Hypothesis
                </div>
                {(() => {
                  const hyp = hypotheses.find((h) => h.id === config.hypothesis_id);
                  if (!hyp) return null;
                  return (
                    <>
                      <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink, marginBottom: 4 }}>{hyp.title}</div>
                      <div style={{ fontSize: 12, color: colors.body, marginBottom: 6 }}>{hyp.statement}</div>
                      <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                        <b>Target:</b> {hyp.target_component} | <b>Status:</b> {hyp.status}
                      </div>
                    </>
                  );
                })()}
              </div>
            )}

            <div style={{ fontSize: 12, color: colors.bodyMuted, padding: 10, backgroundColor: colors.surfacePearl, borderRadius: 6, lineHeight: 1.5 }}>
              Linking an experiment to a hypothesis enables automatic evidence aggregation and hypothesis evaluation upon experiment completion.
            </div>
          </div>
        );

      case 'intervention':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <FieldGroup label="Intervention Type">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {INTERVENTION_TYPES.map((type) => (
                  <label
                    key={type.value}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 8,
                      padding: '8px 10px',
                      borderRadius: 6,
                      border: `1px solid ${config.intervention_type === type.value ? colors.primary : colors.border}`,
                      backgroundColor: config.intervention_type === type.value ? colors.accentSoft : colors.canvas,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <input
                      type="radio"
                      name="intervention_type"
                      value={type.value}
                      checked={config.intervention_type === type.value}
                      onChange={(e) => updateConfig('intervention_type', e.target.value)}
                      style={{ marginTop: 2 }}
                    />
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: colors.ink }}>{type.label}</div>
                      <div style={{ fontSize: 11, color: colors.bodyMuted }}>{type.description}</div>
                    </div>
                  </label>
                ))}
              </div>
            </FieldGroup>

            <FieldGroup label="Target Component">
              <input
                type="text"
                value={config.source_component}
                onChange={(e) => updateConfig('source_component', e.target.value)}
                placeholder="L9H9, L8_MLP, Feature_412"
                style={inputStyle}
              />
            </FieldGroup>

            {config.intervention_type === 'STEERING' && (
              <FieldGroup label="Steering Coefficient">
                <input
                  type="number"
                  step="0.1"
                  value={config.steering_coefficient}
                  onChange={(e) => updateConfig('steering_coefficient', parseFloat(e.target.value) || 1.0)}
                  style={inputStyle}
                />
              </FieldGroup>
            )}

            <FieldGroup label="Number of Trials">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <input
                  type="range"
                  min={1}
                  max={20}
                  value={config.repeats}
                  onChange={(e) => updateConfig('repeats', parseInt(e.target.value, 10))}
                  style={{ flex: 1 }}
                />
                <span style={{ fontSize: 13, fontWeight: 600, color: colors.ink, minWidth: 24, textAlign: 'right' }}>
                  {config.repeats}
                </span>
              </div>
            </FieldGroup>
          </div>
        );

      case 'controls':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <FieldGroup label="Primary Negative Control">
              <input
                type="text"
                value={config.control_component}
                onChange={(e) => updateConfig('control_component', e.target.value)}
                placeholder="L9H8"
                style={inputStyle}
              />
              <div style={{ fontSize: 11, color: colors.bodyMuted, marginTop: 4 }}>
                A component expected to have no effect on the target behavior.
              </div>
            </FieldGroup>

            <FieldGroup label="Additional Control Components">
              {config.control_components.map((ctrl, idx) => (
                <div key={idx} style={{ display: 'flex', gap: 6, marginBottom: 6 }}>
                  <input
                    type="text"
                    value={ctrl}
                    onChange={(e) => updateControlComponent(idx, e.target.value)}
                    placeholder={`Control ${idx + 1}`}
                    style={{ ...inputStyle, flex: 1 }}
                  />
                  <button
                    onClick={() => removeControlComponent(idx)}
                    style={{
                      padding: '4px 8px',
                      borderRadius: 4,
                      border: `1px solid ${colors.dangerBorder}`,
                      backgroundColor: colors.dangerSoft,
                      color: colors.dangerText,
                      fontSize: 11,
                      cursor: 'pointer',
                    }}
                  >
                    Remove
                  </button>
                </div>
              ))}
              <button
                onClick={addControlComponent}
                style={{
                  padding: '6px 12px',
                  borderRadius: 6,
                  border: `1px dashed ${colors.border}`,
                  backgroundColor: 'transparent',
                  color: colors.bodyMuted,
                  fontSize: 12,
                  cursor: 'pointer',
                  textAlign: 'center',
                }}
              >
                + Add Control Component
              </button>
            </FieldGroup>

            <div style={{ fontSize: 12, color: colors.bodyMuted, padding: 10, backgroundColor: colors.surfacePearl, borderRadius: 6, lineHeight: 1.5 }}>
              Controls are used to compute the specificity ratio and establish that the intervention effect is specific to the target component.
            </div>
          </div>
        );

      case 'metrics':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <FieldGroup label="Select Metrics to Compute">
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {AVAILABLE_METRICS.map((metric) => (
                  <label
                    key={metric.id}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 8,
                      padding: '8px 10px',
                      borderRadius: 6,
                      border: `1px solid ${config.selected_metrics.includes(metric.id) ? colors.primary : colors.border}`,
                      backgroundColor: config.selected_metrics.includes(metric.id) ? colors.accentSoft : colors.canvas,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={config.selected_metrics.includes(metric.id)}
                      onChange={() => toggleMetric(metric.id)}
                      style={{ marginTop: 2 }}
                    />
                    <div>
                      <div style={{ fontSize: 12, fontWeight: 600, color: colors.ink }}>{metric.label}</div>
                      <div style={{ fontSize: 11, color: colors.bodyMuted }}>{metric.description}</div>
                    </div>
                  </label>
                ))}
              </div>
            </FieldGroup>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            Experiment Configuration
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Experiment Builder
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Configure and run causal intervention experiments against live model weights.
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={handleReset}
            style={{
              padding: '8px 12px',
              borderRadius: 6,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.bodyMuted,
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <RotateCcw size={13} /> Reset
          </button>
          <button
            onClick={handleRun}
            disabled={running || !config.name.trim()}
            style={{
              padding: '8px 16px',
              borderRadius: 6,
              border: 'none',
              backgroundColor: running || !config.name.trim() ? colors.bodyMuted : colors.primary,
              color: colors.onPrimary,
              fontSize: 12,
              fontWeight: 700,
              cursor: running || !config.name.trim() ? 'default' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            {running ? (
              <>Executing...</>
            ) : (
              <>
                <Play size={14} /> Run Experiment
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {(error || storeError) && (
        <div style={{
          padding: '10px 14px',
          borderRadius: 8,
          backgroundColor: colors.dangerSoft,
          border: `1px solid ${colors.dangerBorder}`,
          color: colors.dangerText,
          fontSize: 12,
          marginBottom: 16,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <AlertCircle size={16} /> {error || storeError}
        </div>
      )}

      {/* Main Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: 16 }}>
        {/* Tab Navigation */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 12px',
                borderRadius: 6,
                border: 'none',
                backgroundColor: activeTab === tab.id ? colors.primary : 'transparent',
                color: activeTab === tab.id ? colors.onPrimary : colors.bodyMuted,
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.15s ease',
              }}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1, minHeight: 400 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: colors.ink, marginBottom: 16, textTransform: 'uppercase', letterSpacing: 0.4, display: 'flex', alignItems: 'center', gap: 8 }}>
            {TABS.find((t) => t.id === activeTab)?.icon}
            {TABS.find((t) => t.id === activeTab)?.label} Configuration
          </div>
          {renderTabContent()}
        </div>
      </div>

      {/* Result Display */}
      {result && (
        <div style={{ marginTop: 16, border: `1px solid ${result.execution_status === 'FAILED' || result.execution_status === 'NOT_EXECUTABLE' ? colors.dangerBorder : colors.successBorder}`, borderRadius: 10, padding: 16, backgroundColor: result.execution_status === 'FAILED' || result.execution_status === 'NOT_EXECUTABLE' ? colors.dangerSoft : colors.successSoft }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            {result.execution_status === 'FAILED' || result.execution_status === 'NOT_EXECUTABLE' ? <XCircle size={16} style={{ color: colors.danger }} /> : <CheckCircle2 size={16} style={{ color: colors.success }} />}
            <span style={{ fontSize: 13, fontWeight: 700, color: result.execution_status === 'FAILED' || result.execution_status === 'NOT_EXECUTABLE' ? colors.dangerText : colors.successText }}>
              {result.execution_status === 'FAILED' ? 'Experiment Failed' : result.execution_status === 'NOT_EXECUTABLE' ? 'Experiment Not Executable' : 'Experiment Completed'}
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
            <MetricCard label="Δ Logit" value={result.delta_logit?.toFixed(3) ?? 'N/A'} />
            <MetricCard label="Baseline Prob" value={`${((result.baseline_target_prob ?? 0) * 100).toFixed(1)}%`} />
            <MetricCard label="Intervened Prob" value={`${((result.intervened_target_prob ?? 0) * 100).toFixed(1)}%`} />
            <MetricCard label="Effect Size" value={result.effect_size_cohens_d?.toFixed(2) ?? 'N/A'} />
          </div>
        </div>
      )}
    </div>
  );
};

/* ── Helper Components ──────────────────────────────────────────────── */

const FieldGroup: React.FC<{ label: string; required?: boolean; children: React.ReactNode }> = ({
  label,
  required,
  children,
}) => (
  <div>
    <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
      {label} {required && <span style={{ color: colors.danger }}>*</span>}
    </label>
    {children}
  </div>
);

const MetricCard: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={{ padding: 10, borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
    <div style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase', marginBottom: 2 }}>{label}</div>
    <div style={{ fontSize: 16, fontWeight: 800, color: colors.ink, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
  </div>
);

/* ── Shared Styles ──────────────────────────────────────────────────── */

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '7px 10px',
  borderRadius: 6,
  border: `1px solid ${colors.border}`,
  backgroundColor: colors.canvas,
  color: colors.ink,
  fontSize: 12,
  boxSizing: 'border-box',
  outline: 'none',
};

const selectStyle: React.CSSProperties = {
  width: '100%',
  padding: '7px 10px',
  borderRadius: 6,
  border: `1px solid ${colors.border}`,
  backgroundColor: colors.canvas,
  color: colors.ink,
  fontSize: 12,
  boxSizing: 'border-box',
};

const cardStyle: React.CSSProperties = {
  padding: 12,
  borderRadius: 8,
  backgroundColor: colors.surfacePearl,
  border: `1px solid ${colors.border}`,
};
