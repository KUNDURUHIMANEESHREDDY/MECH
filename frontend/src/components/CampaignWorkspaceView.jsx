import React, { useState } from 'react';
import { BarChart3, Building2, Check, FlaskConical, ScrollText, TrendingUp, X } from 'lucide-react';
import { colors, radii, spacing, typography } from '../design/tokens';

/**
 * CampaignWorkspaceView - Scientist's Research Campaign Workspace
 *
 * Provides a 5-panel workspace for managing ongoing & completed research campaigns:
 * 1. Overview (Goals, FLOPs, Budget $, Status)
 * 2. Experiments (Completed vs Failed Matrix)
 * 3. Evidence Stream (Accumulated evidence items)
 * 4. Belief Evolution (Bayesian Prior -> Likelihood -> Posterior -> 95% Credible Interval)
 * 5. Mechanism Claims (Emitted scientific claims)
 */

const MOCK_CAMPAIGNS = [
  {
    id: 'campaign_ioi_01',
    title: 'IOI Circuit Discovery & Cross-Model Validation',
    goal: 'Discover, prune, and validate indirect object identification circuit across GPT-2, Gemma, and Llama.',
    status: 'Completed',
    model: 'GPT2-S',
    remaining_uncertainty: 0.038,
    budget_remaining: 8.45,
    initial_budget: 10.00,
    compute_flops: '2.07e13',
    papers_reproduced: ['Wang et al. 2022', 'Conmy et al. 2023'],
    claims: ['IOI Name Mover Circuit'],
    completed_experiments: [
      { id: 'exp_01_attr', name: 'Attribution Patching', model: 'GPT2-S', runtime: '1.2s', status: 'Completed', unc_before: 0.50, unc_after: 0.35, detail: 'Screened 144 heads; isolated L9H9 & L10H0' },
      { id: 'exp_02_acdc', name: 'ACDC Edge Pruning', model: 'GPT2-S', runtime: '7.8s', status: 'Completed', unc_before: 0.35, unc_after: 0.18, detail: 'Pruned 94% of edges; 3-head minimal subgraph' },
      { id: 'exp_03_scrub', name: 'Causal Scrubbing', model: 'GPT2-S', runtime: '9.5s', status: 'Completed', unc_before: 0.18, unc_after: 0.038, detail: 'Preserved 96% accuracy under name-type equivalence class' }
    ],
    failed_experiments: [],
    evidence: [
      { id: 'ev_01', type: 'ACDC Pruning', text: '94% pruned; 3-head minimal subgraph recovers 97.2% logit diff' },
      { id: 'ev_02', type: 'Causal Scrubbing', text: 'Preserved 96% accuracy under name-type equivalence class' },
      { id: 'ev_03', type: 'Universality', text: 'GPT-2 L9H9 aligned with Gemma L11H4 (0.88 cosine similarity)' }
    ],
    belief_history: [
      { iter: 1, alg: 'Prior Initialization', prior: 0.50, likelihood: 0.50, posterior: 0.50, delta: 0.00, ci: '[0.20, 0.80]', verdict: 'NEUTRAL' },
      { iter: 2, alg: 'Attribution Patching', prior: 0.50, likelihood: 0.85, posterior: 0.68, delta: +0.18, ci: '[0.42, 0.86]', verdict: 'SUPPORTED' },
      { iter: 3, alg: 'ACDC Edge Pruning', prior: 0.68, likelihood: 0.90, posterior: 0.82, delta: +0.14, ci: '[0.65, 0.93]', verdict: 'SUPPORTED' },
      { iter: 4, alg: 'Causal Scrubbing', prior: 0.82, likelihood: 0.95, posterior: 0.96, delta: +0.14, ci: '[0.89, 0.99]', verdict: 'SUPPORTED' },
    ]
  },
  {
    id: 'campaign_induction_02',
    title: 'Induction Head Sequence Repeater Study',
    goal: 'Isolate prefix-matching and token-copying induction heads in Gemma-2B.',
    status: 'Running',
    model: 'Gemma-2B',
    remaining_uncertainty: 0.125,
    budget_remaining: 5.80,
    initial_budget: 10.00,
    compute_flops: '4.85e13',
    papers_reproduced: ['Olsson et al. 2022'],
    claims: ['Induction Head Sequence Repeater'],
    completed_experiments: [
      { id: 'exp_01_attr', name: 'Attribution Patching', model: 'Gemma-2B', runtime: '2.1s', status: 'Completed', unc_before: 0.50, unc_after: 0.32, detail: 'High attribution on L5H1 during prefix token repetition' },
      { id: 'exp_02_acdc', name: 'ACDC Edge Pruning', model: 'Gemma-2B', runtime: '12.4s', status: 'Completed', unc_before: 0.32, unc_after: 0.125, detail: 'Minimal 2-head circuit recovers 94% induction accuracy' }
    ],
    failed_experiments: [
      { id: 'exp_fail_01', name: 'Naive Random Resampling', model: 'Gemma-2B', runtime: '3.4s', status: 'Failed', error: 'Convergence error: activation variance exceeded threshold' }
    ],
    evidence: [
      { id: 'ev_10', type: 'Attribution', text: 'L5H1 isolated with 0.93 attribution score' },
      { id: 'ev_11', type: 'ACDC', text: '2-head minimal circuit recovers 94% accuracy' }
    ],
    belief_history: [
      { iter: 1, alg: 'Prior Initialization', prior: 0.50, likelihood: 0.50, posterior: 0.50, delta: 0.00, ci: '[0.20, 0.80]', verdict: 'NEUTRAL' },
      { iter: 2, alg: 'Attribution Patching', prior: 0.50, likelihood: 0.85, posterior: 0.68, delta: +0.18, ci: '[0.42, 0.86]', verdict: 'SUPPORTED' },
      { iter: 3, alg: 'ACDC Edge Pruning', prior: 0.68, likelihood: 0.90, posterior: 0.875, delta: +0.195, ci: '[0.72, 0.95]', verdict: 'SUPPORTED' },
    ]
  }
];

export default function CampaignWorkspaceView({ api, onNavigate }) {
  const [selectedCampaignId, setSelectedCampaignId] = useState(MOCK_CAMPAIGNS[0].id);
  const [activeTab, setActiveTab] = useState('belief'); // overview, experiments, evidence, belief, claims

  const campaign = MOCK_CAMPAIGNS.find(c => c.id === selectedCampaignId) || MOCK_CAMPAIGNS[0];

  return (
    <div style={{ padding: 28, background: colors.canvasParchment, color: colors.ink, height: '100%', overflowY: 'auto', fontFamily: typography.body.fontFamily }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 6px 0', color: colors.purpleBorder, display: 'flex', alignItems: 'center', gap: 8 }}><FlaskConical size={20} /> Scientist's Campaign Workspace</h1>
          <p style={{ margin: 0, color: colors.inkMuted48, fontSize: 13 }}>
            Manage ongoing & completed research campaigns, Bayesian belief evolution, compute FLOPs, and USD budget
          </p>
        </div>
        <div style={{ background: colors.surfacePearl, padding: '6px 14px', borderRadius: 8, border: `1px solid ${colors.hairline}`, fontSize: 12 }}>
          <span style={{ color: colors.inkMuted48 }}>Active Workspace: </span>
          <span style={{ color: colors.primary, fontWeight: 600 }}>{MOCK_CAMPAIGNS.length} Campaigns</span>
        </div>
      </div>

      {/* Campaign Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        {MOCK_CAMPAIGNS.map(c => (
          <button
            key={c.id}
            onClick={() => setSelectedCampaignId(c.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
              background: selectedCampaignId === c.id ? colors.surfacePearl : colors.canvas,
              color: selectedCampaignId === c.id ? colors.purpleBorder : colors.inkMuted48,
              transition: 'all 0.2s'
            }}
          >
            {c.title} ({c.status})
          </button>
        ))}
      </div>

      {/* Workspace Panel Selector Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: `1px solid ${colors.hairline}`, paddingBottom: 10 }}>
        {[
          { key: 'overview', label: 'Executive Overview', icon: BarChart3 },
          { key: 'experiments', label: 'Executed Matrix', icon: FlaskConical },
          { key: 'evidence', label: 'Evidence Stream', icon: ScrollText },
          { key: 'belief', label: 'Bayesian Belief Evolution', icon: TrendingUp },
          { key: 'claims', label: 'Emitted Claims', icon: Building2 },
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
              background: activeTab === tab.key ? colors.surfacePearl : 'transparent',
              color: activeTab === tab.key ? colors.primary : colors.inkMuted48,
              borderBottom: activeTab === tab.key ? `2px solid ${colors.primary}` : '2px solid transparent'
            }}
          >
            {tab.icon ? <tab.icon size={13} style={{ verticalAlign: 'middle', marginRight: 6 }} /> : null}{tab.label}
          </button>
        ))}
      </div>

      {/* Campaign Executive Header Card */}
      <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 20, marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <div>
            <span style={{ fontSize: 11, color: colors.primary, textTransform: 'uppercase', letterSpacing: 1.2, fontWeight: 700 }}>Active Campaign</span>
            <h2 style={{ fontSize: 18, margin: '4px 0 6px 0' }}>{campaign.title}</h2>
            <div style={{ fontSize: 13, color: colors.ink }}>{campaign.goal}</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <span style={{
              background: campaign.status === 'Completed' ? `${colors.success}26` : `${colors.warning}26`,
              color: campaign.status === 'Completed' ? colors.success : colors.warning,
              border: `1px solid ${campaign.status === 'Completed' ? colors.success : colors.warning}`,
              padding: '4px 12px', borderRadius: 6, fontSize: 12, fontWeight: 700
            }}>
              {campaign.status}
            </span>
          </div>
        </div>

        {/* Executive Metrics Bar */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10, background: colors.surfacePearl, padding: 12, borderRadius: 8, border: `1px solid ${colors.hairline}`, textAlign: 'center' }}>
          <div>
            <div style={{ fontSize: 10, color: colors.inkMuted48, textTransform: 'uppercase' }}>Posterior Belief</div>
            <div style={{ fontSize: 16, fontWeight: 800, color: colors.success, marginTop: 2 }}>{((1.0 - campaign.remaining_uncertainty) * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: colors.inkMuted48, textTransform: 'uppercase' }}>Uncertainty Bounds</div>
            <div style={{ fontSize: 16, fontWeight: 800, color: colors.primary, marginTop: 2 }}>{(campaign.remaining_uncertainty * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: colors.inkMuted48, textTransform: 'uppercase' }}>Budget Remaining</div>
            <div style={{ fontSize: 16, fontWeight: 800, color: colors.warning, marginTop: 2 }}>${campaign.budget_remaining.toFixed(2)}</div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: colors.inkMuted48, textTransform: 'uppercase' }}>Compute FLOPs</div>
            <div style={{ fontSize: 16, fontWeight: 800, color: colors.purpleBorder, marginTop: 2 }}>{campaign.compute_flops}</div>
          </div>
          <div>
            <div style={{ fontSize: 10, color: colors.inkMuted48, textTransform: 'uppercase' }}>Papers Reproduced</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: colors.info, marginTop: 4 }}>{campaign.papers_reproduced.length} Papers</div>
          </div>
        </div>
      </div>

      {/* Dynamic Panel Content */}

      {/* PANEL 4: BAYESIAN BELIEF EVOLUTION (Target Phase 30 Panel) */}
      {activeTab === 'belief' && (
        <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 24 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <div>
              <h3 style={{ fontSize: 16, color: colors.purpleBorder, margin: '0 0 4px 0', display: 'flex', alignItems: 'center', gap: 6 }}><TrendingUp size={15} /> Mathematically Traceable Bayesian Belief Evolution</h3>
              <p style={{ fontSize: 12, color: colors.inkMuted48, margin: 0 }}>
                P(Mechanism | Evidence) = [P(Evidence | Mechanism) × Prior] / P(Evidence)
              </p>
            </div>
            <div style={{ background: colors.surfacePearl, padding: '6px 12px', borderRadius: 6, border: `1px solid ${colors.hairline}`, fontSize: 12, color: colors.success, fontWeight: 700 }}>
              Final Posterior Belief: {((1.0 - campaign.remaining_uncertainty) * 100).toFixed(1)}%
            </div>
          </div>

          {/* Belief Evolution Iteration Table */}
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
            <thead>
              <tr style={{ background: colors.surfacePearl, borderBottom: `1px solid ${colors.hairline}`, color: colors.inkMuted48 }}>
                <th style={{ padding: 10 }}>Iter</th>
                <th style={{ padding: 10 }}>Algorithm</th>
                <th style={{ padding: 10 }}>Prior P(M)</th>
                <th style={{ padding: 10 }}>Likelihood P(E|M)</th>
                <th style={{ padding: 10 }}>Posterior P(M|E)</th>
                <th style={{ padding: 10 }}>Δ Belief</th>
                <th style={{ padding: 10 }}>95% Credible Interval</th>
                <th style={{ padding: 10 }}>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {campaign.belief_history.map(row => (
                <tr key={row.iter} style={{ borderBottom: `1px solid ${colors.hairline}`, background: row.iter % 2 === 0 ? colors.canvas : colors.surfacePearl }}>
                  <td style={{ padding: 10, fontWeight: 700, color: colors.purpleBorder }}>{row.iter}</td>
                  <td style={{ padding: 10, fontWeight: 600}}>{row.alg}</td>
                  <td style={{ padding: 10, color: colors.inkMuted48 }}>{(row.prior * 100).toFixed(1)}%</td>
                  <td style={{ padding: 10, color: colors.primary }}>{(row.likelihood * 100).toFixed(1)}%</td>
                  <td style={{ padding: 10, fontWeight: 800, color: colors.success }}>{(row.posterior * 100).toFixed(1)}%</td>
                  <td style={{ padding: 10, fontWeight: 700, color: row.delta >= 0 ? colors.success : colors.danger }}>
                    {row.delta >= 0 ? `+${(row.delta * 100).toFixed(1)}%` : `${(row.delta * 100).toFixed(1)}%`}
                  </td>
                  <td style={{ padding: 10, fontFamily: 'monospace', color: colors.purpleBorder }}>{row.ci}</td>
                  <td style={{ padding: 10 }}>
                    <span style={{
                      background: row.verdict === 'SUPPORTED' ? `${colors.success}26` : colors.surfacePearl,
                      color: row.verdict === 'SUPPORTED' ? colors.success : colors.inkMuted48,
                      padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 700
                    }}>
                      {row.verdict}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* PANEL 2: EXECUTED EXPERIMENTS MATRIX */}
      {activeTab === 'experiments' && (
        <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 20 }}>
          <h3 style={{ fontSize: 14, color: colors.purpleBorder, margin: '0 0 14px 0', display: 'flex', alignItems: 'center', gap: 6 }}><FlaskConical size={14} /> Executed Experiments Matrix</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {campaign.completed_experiments.map(exp => (
              <div key={exp.id} style={{ background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 8, padding: 12, borderLeft: `3px solid ${colors.success}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: 13, fontWeight: 700}}>{exp.name} ({exp.model})</span>
                  <span style={{ fontSize: 11, color: colors.success, fontWeight: 600 }}><Check size={11} style={{ verticalAlign: 'middle' }} /> {exp.status} ({exp.runtime})</span>
                </div>
                <div style={{ fontSize: 12, color: colors.inkMuted48 }}>{exp.detail}</div>
              </div>
            ))}
            {campaign.failed_experiments && campaign.failed_experiments.map(exp => (
              <div key={exp.id} style={{ background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 8, padding: 12, borderLeft: `3px solid ${colors.danger}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: 13, fontWeight: 700}}>{exp.name} ({exp.model})</span>
                  <span style={{ fontSize: 11, color: colors.danger, fontWeight: 600 }}><X size={11} style={{ verticalAlign: 'middle' }} /> {exp.status} ({exp.runtime})</span>
                </div>
                <div style={{ fontSize: 12, color: colors.danger, fontFamily: 'monospace' }}>{exp.error}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* PANEL 3: EVIDENCE STREAM */}
      {activeTab === 'evidence' && (
        <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 20 }}>
          <h3 style={{ fontSize: 14, color: colors.primary, margin: '0 0 12px 0', display: 'flex', alignItems: 'center', gap: 6 }}><ScrollText size={14} /> Accumulated Evidence Stream</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {campaign.evidence.map(ev => (
              <div key={ev.id} style={{ background: colors.canvas, border: `1px solid ${colors.hairline}`, borderRadius: 6, padding: '10px 12px', fontSize: 12 }}>
                <span style={{ color: colors.success, fontWeight: 700, marginRight: 6 }}><Check size={11} style={{ verticalAlign: 'middle' }} /> [{ev.type}]</span>
                <span style={{  }}>{ev.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* PANEL 1: EXECUTIVE OVERVIEW */}
      {activeTab === 'overview' && (
        <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 20 }}>
          <h3 style={{ fontSize: 14, color: colors.purpleBorder, margin: '0 0 12px 0', display: 'flex', alignItems: 'center', gap: 6 }}><BarChart3 size={14} /> Campaign Executive Overview</h3>
          <div style={{ fontSize: 13, color: colors.ink, lineHeight: 1.6 }}>
            This research campaign evaluated target model <strong>{campaign.model}</strong> across {campaign.completed_experiments.length} executed experiment runs.
            The Bayesian belief engine updated posterior belief from <strong>50.0%</strong> to <strong>{((1.0 - campaign.remaining_uncertainty) * 100).toFixed(1)}%</strong>.
          </div>
        </div>
      )}

      {/* PANEL 5: EMITTED CLAIMS */}
      {activeTab === 'claims' && (
        <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 20 }}>
          <h3 style={{ fontSize: 14, color: colors.info, margin: '0 0 12px 0', display: 'flex', alignItems: 'center', gap: 6 }}><Building2 size={14} /> Emitted Mechanism Claims</h3>
          {campaign.claims.map((c, i) => (
            <div key={i} style={{ background: colors.canvas, border: `1px solid ${colors.hairline}`, padding: 12, borderRadius: 8, fontSize: 13, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
              <Building2 size={13} style={{ flexShrink: 0 }} /> {c}
            </div>
          ))}
        </div>
      )}

    </div>
  );
}

