import React, { useState } from 'react';
import { useResearchStore, ExperimentRun } from '../../shared/stores/research';
import { useSelectionStore } from '../../shared/stores/selection';
import { colors } from '../../design/tokens/colors';
import { FlaskConical, Play, ArrowRight, ShieldCheck, AlertCircle, Layers, CheckCircle2 } from 'lucide-react';

export const InterventionLab: React.FC = () => {
  const {
    activeInvestigation,
    activeHypothesis,
    runCausalExperiment,
    selectComponent,
  } = useResearchStore();

  const selection = useSelectionStore();

  const [cleanPrompt, setCleanPrompt] = useState('When Mary and John went to the store, John gave a drink to');
  const [corruptedPrompt, setCorruptedPrompt] = useState('When Mary and John went to the store, Mary gave a drink to');
  const [targetToken, setTargetToken] = useState(' Mary');
  const [distractorToken, setDistractorToken] = useState(' John');

  const [interventionType, setInterventionType] = useState('ACTIVATION_PATCHING');
  const [selectedLayer, setSelectedLayer] = useState(selection.layer ?? 9);
  const [selectedHead, setSelectedHead] = useState(selection.head ?? 9);
  const [isMlp, setIsMlp] = useState(false);
  const [steeringCoeff, setSteeringCoeff] = useState(1.0);
  const [controlComponent, setControlComponent] = useState('L9H8');

  const [running, setRunning] = useState(false);
  const [lastResult, setLastResult] = useState<ExperimentRun | null>(null);
  const [error, setError] = useState<string | null>(null);

  const componentName = isMlp ? `L${selectedLayer}_MLP` : `L${selectedLayer}H${selectedHead}`;

  const handleRun = async () => {
    setRunning(true);
    setError(null);
    try {
      const run = await runCausalExperiment({
        name: `${interventionType} on ${componentName}`,
        clean_prompt: cleanPrompt,
        corrupted_prompt: corruptedPrompt,
        target_token: targetToken,
        distractor_token: distractorToken,
        intervention_type: interventionType,
        source_component: componentName,
        control_component: controlComponent || undefined,
        steering_coefficient: steeringCoeff,
      });
      setLastResult(run);
      // Sync global selection
      if (!isMlp) {
        selection.setSelectedHead(selectedLayer, selectedHead);
        selectComponent({ name: componentName, layer: selectedLayer, head: selectedHead, componentType: 'head' });
      }
    } catch (err: any) {
      setError(err.message || 'Intervention execution failed.');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
          Causal Intervention & Patching Lab
        </div>
        <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
          Execute Controlled Causal Experiments
        </h2>
        <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
          Manipulate internal tensor representations and measure downstream behavioral shifts (ΔLogit, ΔProb).
        </div>
      </div>

      {error && (
        <div style={{ padding: '10px 14px', borderRadius: 8, backgroundColor: colors.dangerSoft, border: `1px solid ${colors.dangerBorder}`, color: colors.dangerText, fontSize: 12, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
          <AlertCircle size={16} /> <b>Execution Error:</b> {error}
        </div>
      )}

      {/* Grid Layout: Config on Left, Results on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 16 }}>
        {/* Configuration Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Prompts Card */}
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14, backgroundColor: colors.surfaceTile1 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 0.4 }}>
              1. Probing Conditions
            </div>

            <div style={{ marginBottom: 10 }}>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Clean (Target) Prompt
              </label>
              <textarea
                value={cleanPrompt}
                onChange={(e) => setCleanPrompt(e.target.value)}
                rows={2}
                style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, outline: 'none', boxSizing: 'border-box' }}
              />
            </div>

            <div style={{ marginBottom: 10 }}>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Corrupted (Source for Patching) Prompt
              </label>
              <textarea
                value={corruptedPrompt}
                onChange={(e) => setCorruptedPrompt(e.target.value)}
                rows={2}
                style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, outline: 'none', boxSizing: 'border-box' }}
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div>
                <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                  Target Token
                </label>
                <input
                  type="text"
                  value={targetToken}
                  onChange={(e) => setTargetToken(e.target.value)}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                  Distractor Token
                </label>
                <input
                  type="text"
                  value={distractorToken}
                  onChange={(e) => setDistractorToken(e.target.value)}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
                />
              </div>
            </div>
          </div>

          {/* Intervention Target Card */}
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14, backgroundColor: colors.surfaceTile1 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 0.4 }}>
              2. Target Component & Intervention Mode
            </div>

            <div style={{ marginBottom: 10 }}>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Intervention Type
              </label>
              <select
                value={interventionType}
                onChange={(e) => setInterventionType(e.target.value)}
                style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12 }}
              >
                <option value="ACTIVATION_PATCHING">Activation Patching (Clean ← Corrupted)</option>
                <option value="ABLATION_ZERO">Zero Ablation (Knockout)</option>
                <option value="ABLATION_MEAN">Mean Ablation</option>
                <option value="STEERING">Activation Steering (Multiplier)</option>
              </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 10 }}>
              <div>
                <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                  Layer (0–11)
                </label>
                <input
                  type="number"
                  min={0}
                  max={11}
                  value={selectedLayer}
                  onChange={(e) => setSelectedLayer(parseInt(e.target.value, 10))}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                  Attention Head (0–11)
                </label>
                <input
                  type="number"
                  min={0}
                  max={11}
                  disabled={isMlp}
                  value={selectedHead}
                  onChange={(e) => setSelectedHead(parseInt(e.target.value, 10))}
                  style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
                />
              </div>
            </div>

            <div style={{ marginBottom: 10 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={isMlp}
                  onChange={(e) => setIsMlp(e.target.checked)}
                />
                <span>Intervene on MLP block output instead of attention head</span>
              </label>
            </div>

            <div style={{ marginBottom: 10 }}>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Negative Control Component (e.g. L9H8, L0H0)
              </label>
              <input
                type="text"
                value={controlComponent}
                onChange={(e) => setControlComponent(e.target.value)}
                placeholder="L9H8"
                style={{ width: '100%', padding: '6px 8px', borderRadius: 6, border: `1px solid ${colors.border}`, backgroundColor: colors.canvas, color: colors.ink, fontSize: 12, boxSizing: 'border-box' }}
              />
            </div>

            <button
              onClick={handleRun}
              disabled={running}
              style={{
                width: '100%',
                padding: '10px 0',
                borderRadius: 8,
                border: 'none',
                backgroundColor: running ? colors.bodyMuted : colors.primary,
                color: colors.onPrimary,
                fontSize: 13,
                fontWeight: 700,
                cursor: running ? 'default' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
              }}
            >
              {running ? <span className="spinner">Executing Live PyTorch Hooks…</span> : <><Play size={15} /> Execute Causal Intervention</>}
            </button>
          </div>
        </div>

        {/* Results Column */}
        <div>
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1, height: '100%', boxSizing: 'border-box' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 0.4 }}>
              Intervention Result & Causal Effect
            </div>

            {lastResult ? (
              <div>
                {/* Metric Summary Cards */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 }}>
                  <div style={{ padding: 12, borderRadius: 8, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                    <div style={{ fontSize: 11, color: colors.bodyMuted, textTransform: 'uppercase' }}>Logit Margin Shift</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: lastResult.delta_logit > 1.0 ? colors.successText : colors.dangerText, fontVariantNumeric: 'tabular-nums' }}>
                      Δ = {lastResult.delta_logit.toFixed(2)}
                    </div>
                    <div style={{ fontSize: 11, color: colors.bodyMuted, marginTop: 4 }}>
                      Clean: {lastResult.baseline_logit.toFixed(2)} → Int: {lastResult.intervened_logit.toFixed(2)}
                    </div>
                  </div>

                  <div style={{ padding: 12, borderRadius: 8, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                    <div style={{ fontSize: 11, color: colors.bodyMuted, textTransform: 'uppercase' }}>Target Probability</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color: colors.ink, fontVariantNumeric: 'tabular-nums' }}>
                      -{(lastResult.delta_target_prob * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: 11, color: colors.bodyMuted, marginTop: 4 }}>
                      {(lastResult.baseline_target_prob * 100).toFixed(1)}% → {(lastResult.intervened_target_prob * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Control Comparison */}
                <div style={{ padding: 10, borderRadius: 8, backgroundColor: colors.surfacePearl, border: `1px solid ${colors.border}`, marginBottom: 14 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase', marginBottom: 4 }}>
                    Control Comparison & Effect Size
                  </div>
                  <div style={{ fontSize: 12, color: colors.ink }}>
                    Control Component (<b>{controlComponent || 'None'}</b>) ΔLogit: <b>{lastResult.control_delta_logit != null ? lastResult.control_delta_logit.toFixed(2) : 'N/A'}</b>
                  </div>
                  <div style={{ fontSize: 12, color: colors.ink, marginTop: 2 }}>
                    Standardized Effect Size (Cohen's d): <b style={{ color: colors.primary }}>{lastResult.effect_size_cohens_d?.toFixed(2) || '0.00'}</b>
                  </div>
                </div>

                {/* Predictions Comparison */}
                <div style={{ marginBottom: 14 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: colors.bodyMuted, textTransform: 'uppercase', marginBottom: 6 }}>
                    Top Predictions (Clean vs Intervened)
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {/* Clean */}
                    <div style={{ padding: 8, borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                      <div style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, marginBottom: 4 }}>BEFORE INTERVENTION</div>
                      {lastResult.top_predicted_tokens_clean.slice(0, 4).map((t, idx) => (
                        <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                          <code>{t.token}</code>
                          <span>{(t.probability * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                    {/* Intervened */}
                    <div style={{ padding: 8, borderRadius: 6, backgroundColor: colors.canvas, border: `1px solid ${colors.border}` }}>
                      <div style={{ fontSize: 10, fontWeight: 700, color: colors.bodyMuted, marginBottom: 4 }}>AFTER INTERVENTION</div>
                      {lastResult.top_predicted_tokens_intervened.slice(0, 4).map((t, idx) => (
                        <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                          <code>{t.token}</code>
                          <span>{(t.probability * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Execution Logs & Provenance */}
                <div style={{ fontSize: 11, color: colors.bodyMuted, backgroundColor: colors.canvas, padding: 8, borderRadius: 6, border: `1px solid ${colors.border}` }}>
                  <div style={{ fontWeight: 600, color: colors.ink, marginBottom: 4 }}>Execution Diagnostics</div>
                  <div>Execution time: {lastResult.execution_time_ms.toFixed(1)}ms</div>
                  <div>Manifest SHA256: <code>{lastResult.provenance_hash?.slice(0, 24)}…</code></div>
                </div>
              </div>
            ) : (
              <div style={{ height: 260, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: colors.bodyMuted, textAlign: 'center' }}>
                <FlaskConical size={36} style={{ opacity: 0.3, marginBottom: 8 }} />
                <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>No Active Intervention Run</div>
                <div style={{ fontSize: 12, maxWidth: 260, marginTop: 4 }}>
                  Configure your probing conditions and target component, then click Execute to measure live causal effects.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
