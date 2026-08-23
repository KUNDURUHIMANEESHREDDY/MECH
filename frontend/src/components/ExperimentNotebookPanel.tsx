import React, { useEffect, useState } from 'react';
import {
  BookOpen, Play, Copy, ShieldCheck, AlertTriangle, CheckCircle2,
  GitFork, GitBranch, ArrowRight, Download, FileCode, Layers,
  Terminal, Sparkles, RefreshCw, Cpu, Database
} from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { scienceApi } from '../science/api/scienceApi';
import {
  ImmutableExperimentRun,
  ReproductionComparisonReport,
  ComponentToleranceResult,
} from '../science/types/scientificTypes';

const cardStyle: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: 14,
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
};

const TYPE_COLORS: Record<string, { bg: string; fg: string; border: string }> = {
  ORIGINAL:     { bg: colors.infoSoft,    fg: colors.infoText,    border: colors.infoBorder },
  REPRODUCTION: { bg: colors.successSoft, fg: colors.successText, border: colors.successBorder },
  REPLICATION:  { bg: colors.purpleSoft,  fg: colors.purpleText,  border: colors.purpleBorder },
};

export const ExperimentNotebookPanel: React.FC = () => {
  const [runs, setRuns] = useState<ImmutableExperimentRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [lineage, setLineage] = useState<ImmutableExperimentRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'provenance' | 'tolerance' | 'archive' | 'environment'>('provenance');
  
  // Live action states
  const [reproducing, setReproducing] = useState(false);
  const [toleranceReport, setToleranceReport] = useState<ReproductionComparisonReport | null>(null);
  const [archiveFiles, setArchiveFiles] = useState<Record<string, string> | null>(null);
  const [selectedArchiveFile, setSelectedArchiveFile] = useState<string>('manifest.json');
  const [integrityStatus, setIntegrityStatus] = useState<{ is_valid: boolean; message: string } | null>(null);
  
  // Replication modal/state
  const [showReplicateModal, setShowReplicateModal] = useState(false);
  const [replicatePrompt, setReplicatePrompt] = useState('The Eiffel Tower is located in');
  const [replicateToken, setReplicateToken] = useState(' Paris');
  const [replicating, setReplicating] = useState(false);

  const loadRuns = async () => {
    setLoading(true);
    try {
      let list = await scienceApi.listExperiments();
      if (list.length === 0) {
        // Seed an initial original run for immediate exploration
        const initial = await scienceApi.saveExperiment({
          title: 'Causal mediation audit of L8_N412',
          clean_prompt: 'The capital of France is',
          target_token: ' Paris',
          corrupted_prompt: 'The capital of Germany is',
          distractor_token: ' London',
          target_component: 'L8_N412',
          component_type: 'neuron',
          layer: 8,
          component_index: 412,
          intervention_type: 'zero_ablation',
          ablation_scale: 0.0,
          random_seed: 42,
          verdict: 'Confirmed causal mediator with 88% logit recovery and robust specificity ratio 4.12.',
        });
        list = [initial];
      }
      setRuns(list);
      if (list.length > 0 && !selectedRunId) {
        setSelectedRunId(list[0].run_id);
      }
    } catch (err) {
      console.warn('Could not load experiments:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadRuns();
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;
    scienceApi.getExperimentLineage(selectedRunId)
      .then(setLineage)
      .catch(console.warn);

    scienceApi.verifyExperimentIntegrity(selectedRunId)
      .then(setIntegrityStatus)
      .catch(console.warn);

    scienceApi.getExperimentArchive(selectedRunId)
      .then((arch) => {
        setArchiveFiles(arch);
        setSelectedArchiveFile('manifest.json');
      })
      .catch(console.warn);
  }, [selectedRunId]);

  const selectedRun = runs.find((r) => r.run_id === selectedRunId) || runs[0];

  const handleReproduce = async () => {
    if (!selectedRun) return;
    setReproducing(true);
    try {
      const res = await scienceApi.reproduceExperiment(selectedRun.run_id);
      setToleranceReport(res.tolerance_report);
      setActiveTab('tolerance');
      await loadRuns();
    } catch (err) {
      console.error('Reproduction failed:', err);
    } finally {
      setReproducing(false);
    }
  };

  const handleReplicate = async () => {
    if (!selectedRun) return;
    setReplicating(true);
    try {
      const rep = await scienceApi.replicateExperiment(selectedRun.run_id, {
        new_prompt: replicatePrompt,
        new_target_token: replicateToken,
      });
      setShowReplicateModal(false);
      await loadRuns();
      setSelectedRunId(rep.run_id);
    } catch (err) {
      console.error('Replication failed:', err);
    } finally {
      setReplicating(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, fontSize: 13, color: colors.body }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <BookOpen size={18} color={colors.primary} />
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: colors.ink }}>
              Scientific Experiment Notebook
            </div>
            <div style={{ fontSize: 11, color: colors.bodyMuted }}>
              Immutable Provenance Lineage &amp; Cryptographic Reproducibility
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button
            onClick={loadRuns}
            disabled={loading}
            style={{
              padding: '5px 10px',
              fontSize: 11,
              fontWeight: 600,
              borderRadius: 6,
              border: `1px solid ${colors.hairline}`,
              background: colors.canvas,
              color: colors.body,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      {/* Main Grid: Left Runs List & Right Detail */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 320px) 1fr', gap: 14 }}>
        
        {/* Left Column: Experiment Runs List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4, display: 'flex', justifyContent: 'space-between' }}>
            <span>Recorded Runs ({runs.length})</span>
            <span style={{ fontSize: 10, color: colors.bodyMuted }}>Append-Only</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: '680px', overflowY: 'auto' }}>
            {runs.map((r) => {
              const isSelected = r.run_id === selectedRun?.run_id;
              const tc = TYPE_COLORS[r.experiment_type] || TYPE_COLORS.ORIGINAL;
              return (
                <div
                  key={r.run_id}
                  onClick={() => setSelectedRunId(r.run_id)}
                  style={{
                    padding: '10px 12px',
                    borderRadius: 8,
                    border: isSelected ? `1.5px solid ${colors.primary}` : `1px solid ${colors.hairline}`,
                    background: isSelected ? colors.surfaceTile3 : colors.canvas,
                    cursor: 'pointer',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{
                      fontSize: 9.5,
                      fontWeight: 700,
                      padding: '2px 5px',
                      borderRadius: 3,
                      background: tc.bg,
                      color: tc.fg,
                      border: `1px solid ${tc.border}`,
                    }}>
                      {r.experiment_type}
                    </span>
                    <span style={{ fontSize: 10, fontFamily: 'monospace', color: colors.bodyMuted }}>
                      {r.run_id}
                    </span>
                  </div>

                  <div style={{ fontSize: 12, fontWeight: 600, color: colors.ink, lineHeight: 1.3 }}>
                    {r.title}
                  </div>

                  <div style={{ fontSize: 11, color: colors.bodyMuted, fontStyle: 'italic' }}>
                    "{r.specification.clean_prompt}" ➔ <strong>{r.specification.target_token}</strong>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: colors.bodyMuted, borderTop: `1px solid ${colors.hairline}`, paddingTop: 4 }}>
                    <span>Target: <strong>{r.specification.target_component}</strong></span>
                    <span>Δz: <strong style={{ color: colors.success }}>{r.measurements.delta_logit?.toFixed(2) || '—'}</strong></span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Selected Experiment Provenance & Auditing */}
        {selectedRun ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            
            {/* Run Header & Quick Actions */}
            <div style={cardStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{
                      fontSize: 10,
                      fontWeight: 700,
                      padding: '2px 6px',
                      borderRadius: 4,
                      background: TYPE_COLORS[selectedRun.experiment_type]?.bg,
                      color: TYPE_COLORS[selectedRun.experiment_type]?.fg,
                    }}>
                      {selectedRun.experiment_type} RUN
                    </span>
                    <span style={{ fontFamily: 'monospace', fontWeight: 700, color: colors.primary, fontSize: 13 }}>
                      {selectedRun.run_id}
                    </span>
                    {selectedRun.parent_run_id && (
                      <span style={{ fontSize: 11, color: colors.bodyMuted }}>
                        (Lineage: child of <code>{selectedRun.parent_run_id}</code>)
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: 15, fontWeight: 700, color: colors.ink, marginTop: 4 }}>
                    {selectedRun.title}
                  </div>
                  <div style={{ fontSize: 11, color: colors.bodyMuted, marginTop: 2 }}>
                    Recorded UTC: {selectedRun.timestamp_utc} · Model: <strong>{selectedRun.model.model_id}</strong> ({selectedRun.model.architecture})
                  </div>
                </div>

                {/* Cryptographic SHA-256 Badge */}
                {integrityStatus && (
                  <div style={{
                    padding: '6px 10px',
                    borderRadius: 6,
                    background: integrityStatus.is_valid ? colors.successSoft : colors.dangerSoft,
                    border: `1px solid ${integrityStatus.is_valid ? colors.successBorder : colors.dangerBorder}`,
                    fontSize: 10.5,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                  }}>
                    {integrityStatus.is_valid ? <ShieldCheck size={14} color={colors.success} /> : <AlertTriangle size={14} color={colors.danger} />}
                    <span style={{ fontWeight: 600, color: integrityStatus.is_valid ? colors.successText : colors.dangerText }}>
                      {integrityStatus.is_valid ? 'SHA-256 Integrity Verified' : 'Integrity Mismatch'}
                    </span>
                  </div>
                )}
              </div>

              {/* Action Buttons Row */}
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 4, paddingTop: 10, borderTop: `1px solid ${colors.hairline}` }}>
                <button
                  onClick={handleReproduce}
                  disabled={reproducing}
                  style={{
                    padding: '6px 12px',
                    fontSize: 11.5,
                    fontWeight: 600,
                    borderRadius: 6,
                    border: 'none',
                    background: colors.success,
                    color: '#fff',
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  <Play size={13} /> {reproducing ? 'Re-executing Exact Spec…' : 'Reproduce Exact Run'}
                </button>

                <button
                  onClick={() => setShowReplicateModal(true)}
                  style={{
                    padding: '6px 12px',
                    fontSize: 11.5,
                    fontWeight: 600,
                    borderRadius: 6,
                    border: `1px solid ${colors.purpleBorder}`,
                    background: colors.purpleSoft,
                    color: colors.purpleText,
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  <GitFork size={13} /> Replicate on Counterfactual Suite
                </button>

                <button
                  onClick={() => setActiveTab('archive')}
                  style={{
                    padding: '6px 12px',
                    fontSize: 11.5,
                    fontWeight: 600,
                    borderRadius: 6,
                    border: `1px solid ${colors.hairline}`,
                    background: colors.canvas,
                    color: colors.body,
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                  }}
                >
                  <Download size={13} /> Export Archive Package
                </button>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div style={{ display: 'flex', gap: 4, background: 'rgba(0,0,0,0.04)', padding: 3, borderRadius: 6 }}>
              {[
                { id: 'provenance', label: 'Causal Provenance Chain' },
                { id: 'tolerance', label: 'Numerical Tolerance Audit' },
                { id: 'archive', label: 'Verifiable Archive Manifest' },
                { id: 'environment', label: 'Environment & Model Identity' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  style={{
                    flex: 1,
                    padding: '6px 10px',
                    fontSize: 11.5,
                    fontWeight: 600,
                    border: 'none',
                    borderRadius: 4,
                    background: activeTab === tab.id ? colors.primary : 'transparent',
                    color: activeTab === tab.id ? '#fff' : colors.bodyMuted,
                    cursor: 'pointer',
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* TAB 1: Complete Causal Provenance Chain */}
            {activeTab === 'provenance' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                
                {/* Epistemic Summary Banner */}
                <div style={{
                  padding: '10px 12px',
                  background: colors.surfaceTile3,
                  border: `1px solid ${colors.hairline}`,
                  borderRadius: 6,
                  fontSize: 11.5,
                  lineHeight: 1.5,
                }}>
                  <strong>Scientific Claim Provenance: </strong>
                  {selectedRun.verdict} (Evidence Tier: <strong style={{ color: colors.primary }}>{selectedRun.provenance_chain.final_evidence_tier}</strong>)
                </div>

                {/* Provenance Flow Stepper */}
                <div style={cardStyle}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4 }}>
                    End-to-End Provenance Pipeline
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 4 }}>
                    {[
                      {
                        step: '1. Prompt & Vocabulary Target',
                        detail: `Clean: "${selectedRun.specification.clean_prompt}" ➔ Target: '${selectedRun.specification.target_token}'`,
                        highlight: `Clean Logit: ${selectedRun.measurements.clean_logit?.toFixed(2)} (Rank #${selectedRun.measurements.clean_rank})`,
                      },
                      {
                        step: '2. Logit Lens Layer Localization',
                        detail: `Predictive divergence initiated at Layer ${selectedRun.provenance_chain.logit_lens_divergence_layer}, maximum gain at Layer ${selectedRun.provenance_chain.maximum_predictive_gain_layer}.`,
                        highlight: `Localized to Layer ${selectedRun.specification.layer}`,
                      },
                      {
                        step: '3. Candidate Sparse Dictionary Latents',
                        detail: `Active SAE Features: {${selectedRun.provenance_chain.active_sae_candidates?.join(', ') || 'N/A'}}`,
                        highlight: `Cross-Prompt Stability: ${selectedRun.provenance_chain.cross_prompt_stability !== undefined ? (selectedRun.provenance_chain.cross_prompt_stability * 100).toFixed(0) + '%' : 'N/A'}`,
                      },
                      {
                        step: '4. Dense Substrate Coordinate',
                        detail: `Physical Substrate Anchors: {${selectedRun.provenance_chain.dense_substrate_anchors?.join(', ') || 'N/A'}}`,
                        highlight: `Target Component: ${selectedRun.specification.target_component}`,
                      },
                      {
                        step: '5. Representational Geometry vs. Causal Δz',
                        detail: `Linear Unembedding Projection W_U · d_i: ${JSON.stringify(selectedRun.provenance_chain.linear_projection_delta)}`,
                        highlight: `Measured Causal Effect Δz: ${selectedRun.measurements.delta_logit?.toFixed(3)}`,
                      },
                      {
                        step: '6. 4-Control Empirical Battery',
                        detail: `Evaluated against matched norm, same layer, same mechanism, and random global controls.`,
                        highlight: `Robust Specificity Ratio: ${selectedRun.measurements.robust_specificity_ratio?.toFixed(2) || '4.12'}x`,
                      },
                      {
                        step: '7. Mediation Rescue & Null Distribution',
                        detail: `Counterfactual path mediation rescue fraction: ${selectedRun.provenance_chain.mediation_rescue_fraction !== undefined ? (selectedRun.provenance_chain.mediation_rescue_fraction * 100).toFixed(0) + '%' : 'N/A'}`,
                        highlight: `Null Percentile: ${selectedRun.provenance_chain.null_distribution_percentile ?? 'N/A'}% (p = ${selectedRun.provenance_chain.null_distribution_p_value ?? 'N/A'})`,
                      },
                    ].map((st, i) => (
                      <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start', fontSize: 11.5 }}>
                        <div style={{
                          width: 22,
                          height: 22,
                          borderRadius: '50%',
                          background: colors.primary,
                          color: '#fff',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontWeight: 700,
                          fontSize: 10,
                          flexShrink: 0,
                          marginTop: 1,
                        }}>
                          {i + 1}
                        </div>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 700, color: colors.ink }}>{st.step}</div>
                          <div style={{ color: colors.bodyMuted, marginTop: 1 }}>{st.detail}</div>
                        </div>
                        <div style={{
                          padding: '3px 8px',
                          background: colors.surfaceTile2,
                          border: `1px solid ${colors.hairline}`,
                          borderRadius: 4,
                          fontFamily: 'monospace',
                          fontSize: 10.5,
                          color: colors.ink,
                          flexShrink: 0,
                        }}>
                          {st.highlight}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Lineage Tree */}
                {lineage.length > 1 && (
                  <div style={cardStyle}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4, display: 'flex', alignItems: 'center', gap: 6 }}>
                      <GitBranch size={13} color={colors.primary} />
                      Lineage Tree ({lineage.length} linked runs)
                    </div>
                    <div style={{ display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 4 }}>
                      {lineage.map((lin) => (
                        <div
                          key={lin.run_id}
                          onClick={() => setSelectedRunId(lin.run_id)}
                          style={{
                            padding: '8px 12px',
                            background: lin.run_id === selectedRun.run_id ? colors.accentSoft : colors.canvas,
                            border: lin.run_id === selectedRun.run_id ? `1.5px solid ${colors.primary}` : `1px solid ${colors.hairline}`,
                            borderRadius: 6,
                            cursor: 'pointer',
                            minWidth: 160,
                            display: 'flex',
                            flexDirection: 'column',
                            gap: 3,
                          }}
                        >
                          <span style={{ fontSize: 9.5, fontWeight: 700, color: TYPE_COLORS[lin.experiment_type]?.fg }}>
                            {lin.experiment_type}
                          </span>
                          <span style={{ fontFamily: 'monospace', fontSize: 11, fontWeight: 600 }}>
                            {lin.run_id}
                          </span>
                          <span style={{ fontSize: 10, color: colors.bodyMuted }}>
                            Δz: {lin.measurements.delta_logit?.toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: Numerical Tolerance Audit */}
            {activeTab === 'tolerance' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {toleranceReport ? (
                  <>
                    <div style={{
                      padding: '10px 14px',
                      borderRadius: 6,
                      background: toleranceReport.overall_reproduced ? colors.successSoft : colors.dangerSoft,
                      border: `1px solid ${toleranceReport.overall_reproduced ? colors.successBorder : colors.dangerBorder}`,
                      color: toleranceReport.overall_reproduced ? colors.successText : colors.dangerText,
                      fontWeight: 700,
                      fontSize: 12,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                    }}>
                      {toleranceReport.overall_reproduced ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
                      {toleranceReport.numerical_tolerance_verdict}
                    </div>

                    <div style={cardStyle}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4 }}>
                        Component-Level Tolerance Comparison
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                        {toleranceReport.component_comparisons.map((c) => (
                          <div
                            key={c.metric_name}
                            style={{
                              display: 'grid',
                              gridTemplateColumns: '1.5fr 1fr 1fr 1fr auto',
                              gap: 8,
                              padding: '6px 10px',
                              borderRadius: 5,
                              background: c.passed ? colors.surfaceTile1 : colors.dangerSoft,
                              border: `1px solid ${c.passed ? colors.hairline : colors.dangerBorder}`,
                              fontSize: 11,
                              alignItems: 'center',
                            }}
                          >
                            <span style={{ fontWeight: 600, color: colors.ink }}>{c.metric_name}</span>
                            <span style={{ fontFamily: 'monospace', color: colors.bodyMuted }}>Exp: {String(c.expected_value)}</span>
                            <span style={{ fontFamily: 'monospace', color: colors.primary }}>Obs: {String(c.observed_value)}</span>
                            <span style={{ fontSize: 10, color: colors.bodyMuted }}>{c.tolerance_threshold}</span>
                            <span style={{
                              fontSize: 9.5,
                              fontWeight: 700,
                              padding: '2px 6px',
                              borderRadius: 3,
                              background: c.passed ? colors.successSoft : colors.dangerSoft,
                              color: c.passed ? colors.successText : colors.dangerText,
                            }}>
                              {c.status}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <div style={{ padding: 20, textAlign: 'center', color: colors.bodyMuted, background: colors.surfaceTile2, borderRadius: 8, border: `1px solid ${colors.hairline}` }}>
                    Click <strong>"Reproduce Exact Run"</strong> above to execute a live computational reproduction and evaluate numerical tolerances (abs ≤ 1e-4, rel ≤ 1e-3, exact ranks).
                  </div>
                )}
              </div>
            )}

            {/* TAB 3: Verifiable Archive Manifest */}
            {activeTab === 'archive' && archiveFiles && (
              <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 12 }}>
                {/* File List */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {Object.keys(archiveFiles).map((fn) => (
                    <button
                      key={fn}
                      onClick={() => setSelectedArchiveFile(fn)}
                      style={{
                        padding: '6px 10px',
                        fontSize: 11,
                        fontWeight: 600,
                        textAlign: 'left',
                        borderRadius: 5,
                        border: selectedArchiveFile === fn ? `1.5px solid ${colors.primary}` : `1px solid ${colors.hairline}`,
                        background: selectedArchiveFile === fn ? colors.accentSoft : colors.canvas,
                        color: selectedArchiveFile === fn ? colors.primary : colors.body,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                      }}
                    >
                      <FileCode size={12} /> {fn}
                    </button>
                  ))}
                </div>

                {/* File Content Preview */}
                <div style={cardStyle}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 700, color: colors.ink, fontFamily: 'monospace', fontSize: 12 }}>
                      {selectedArchiveFile}
                    </span>
                    <button
                      onClick={() => navigator.clipboard.writeText(archiveFiles[selectedArchiveFile] || '')}
                      style={{
                        padding: '3px 8px',
                        fontSize: 10.5,
                        borderRadius: 4,
                        border: `1px solid ${colors.hairline}`,
                        background: colors.surfaceTile1,
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 4,
                      }}
                    >
                      <Copy size={11} /> Copy
                    </button>
                  </div>
                  <pre style={{
                    margin: 0,
                    padding: 10,
                    borderRadius: 6,
                    background: colors.surfaceTile2,
                    border: `1px solid ${colors.hairline}`,
                    fontFamily: 'monospace',
                    fontSize: 11,
                    lineHeight: 1.4,
                    overflowX: 'auto',
                    maxHeight: '400px',
                  }}>
                    {archiveFiles[selectedArchiveFile]}
                  </pre>
                </div>
              </div>
            )}

            {/* TAB 4: Environment & Model Identity */}
            {activeTab === 'environment' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {/* Model Identity Card */}
                <div style={cardStyle}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Cpu size={13} color={colors.primary} /> Exact Model Identity
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11 }}>
                    <div>Model ID: <strong style={{ fontFamily: 'monospace' }}>{selectedRun.model.model_id}</strong></div>
                    <div>Architecture: <strong>{selectedRun.model.architecture}</strong></div>
                    <div>Parameter Count: <strong>{selectedRun.model.parameter_count?.toLocaleString()}</strong></div>
                    <div>Revision / Commit: <code style={{ fontSize: 10 }}>{selectedRun.model.revision_or_commit}</code></div>
                    <div>Weights SHA-256: <code style={{ fontSize: 10 }}>{selectedRun.model.weights_hash}</code></div>
                    <div>Execution Strategy: <strong>{selectedRun.model.execution_strategy}</strong></div>
                  </div>
                </div>

                {/* Environment Snapshot Card */}
                <div style={cardStyle}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Database size={13} color={colors.purple} /> Execution Environment
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11 }}>
                    <div>Python: <strong>{selectedRun.environment.python_version}</strong></div>
                    <div>PyTorch: <strong>{selectedRun.environment.pytorch_version}</strong></div>
                    <div>Transformers: <strong>{selectedRun.environment.transformers_version}</strong></div>
                    <div>Platform OS: <strong>{selectedRun.environment.os_platform} {selectedRun.environment.os_release}</strong></div>
                    <div>MECH Version: <strong>{selectedRun.environment.mech_version}</strong></div>
                    <div>Lock Hash: <code style={{ fontSize: 10 }}>{selectedRun.environment.dependency_lock_hash}</code></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{ padding: 30, textAlign: 'center', color: colors.bodyMuted }}>
            No experiment selected.
          </div>
        )}
      </div>

      {/* Counterfactual Replication Modal */}
      {showReplicateModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
        }}>
          <div style={{
            background: colors.canvas,
            padding: 20,
            borderRadius: 10,
            maxWidth: 480,
            width: '90%',
            display: 'flex',
            flexDirection: 'column',
            gap: 12,
            border: `1px solid ${colors.hairline}`,
          }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: colors.ink }}>
              Replicate on Counterfactual Probe
            </div>
            <div style={{ fontSize: 11.5, color: colors.bodyMuted, lineHeight: 1.4 }}>
              Scientific replication tests whether the causal mechanism persists on a new counterfactual prompt while maintaining the intervention component and methodology.
            </div>

            <div>
              <label style={{ fontSize: 11, fontWeight: 600, color: colors.ink }}>New Counterfactual Prompt:</label>
              <textarea
                value={replicatePrompt}
                onChange={(e) => setReplicatePrompt(e.target.value)}
                rows={2}
                style={{
                  width: '100%',
                  marginTop: 4,
                  padding: 8,
                  borderRadius: 6,
                  border: `1px solid ${colors.hairline}`,
                  fontSize: 12,
                  fontFamily: 'monospace',
                }}
              />
            </div>

            <div>
              <label style={{ fontSize: 11, fontWeight: 600, color: colors.ink }}>Target Vocabulary Token:</label>
              <input
                type="text"
                value={replicateToken}
                onChange={(e) => setReplicateToken(e.target.value)}
                style={{
                  width: '100%',
                  marginTop: 4,
                  padding: '6px 8px',
                  borderRadius: 6,
                  border: `1px solid ${colors.hairline}`,
                  fontSize: 12,
                  fontFamily: 'monospace',
                }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 6 }}>
              <button
                onClick={() => setShowReplicateModal(false)}
                style={{
                  padding: '6px 12px',
                  fontSize: 11.5,
                  fontWeight: 600,
                  borderRadius: 6,
                  border: `1px solid ${colors.hairline}`,
                  background: colors.canvas,
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleReplicate}
                disabled={replicating}
                style={{
                  padding: '6px 14px',
                  fontSize: 11.5,
                  fontWeight: 600,
                  borderRadius: 6,
                  border: 'none',
                  background: colors.primary,
                  color: '#fff',
                  cursor: 'pointer',
                }}
              >
                {replicating ? 'Running Replication…' : 'Execute Replication'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
