import React, { useState } from 'react';
import { ArrowRight, BarChart3, Lightbulb, Star, TriangleAlert, Trophy, Zap } from 'lucide-react';
import { colors, radii, spacing, typography } from '../design/tokens';

/**
 * ResearchAnalyticsView - Multi-Campaign Meta-Learning & Analytics Dashboard
 *
 * Displays empirical meta-learning insights across all historical research campaigns:
 * Algorithm Leaderboard, Best Experiment Sequences, Failure Mode Analysis, and Compute Efficiency.
 */

const MOCK_ANALYTICS = {
  total_campaigns: 150,
  overall_success_rate: 0.964,
  total_flops: '1.45e15',
  total_usd: 84.50,
  leaderboard: [
    { name: 'Causal Scrubbing', success: '98.2%', avg_gain: '+0.18', runtime: '9.5s', gain_tflop: '0.182', gain_usd: '$3.64' },
    { name: 'ACDC Edge Pruning', success: '94.5%', avg_gain: '+0.14', runtime: '7.8s', gain_tflop: '0.165', gain_usd: '$3.30' },
    { name: 'Attribution Patching', success: '99.1%', avg_gain: '+0.18', runtime: '1.2s', gain_tflop: '0.450', gain_usd: '$9.00' },
    { name: 'Feature Universality', success: '91.2%', avg_gain: '+0.12', runtime: '6.2s', gain_tflop: '0.140', gain_usd: '$2.80' },
    { name: 'Transcoder Learning', success: '92.4%', avg_gain: '+0.15', runtime: '5.4s', gain_tflop: '0.155', gain_usd: '$3.10' },
  ],
  sequences: [
    { rank: 1, name: 'IOI Standard Circuit Discovery', seq: ['Attribution Patching', 'ACDC Pruning', 'Causal Scrubbing', 'Transcoders'], posterior: '96.2%', occurrences: 42, speed: '95%' },
    { rank: 2, name: 'Induction Cross-Model Generalization', seq: ['Attribution Patching', 'ACDC Pruning', 'Feature Universality'], posterior: '94.1%', occurrences: 28, speed: '91%' },
  ],
  failures: [
    { id: 'f1', title: 'Feature Universality before ACDC', count: 22, cause: 'Missing underlying causal subgraph structure prior to cross-model matching.', remedy: 'Always schedule ACDC or Path Patching before executing Feature Universality.' },
    { id: 'f2', title: 'Naive Random Activation Resampling', count: 9, cause: 'Unconstrained random activations break input distribution manifold.', remedy: 'Enforce equivalence-class resamplings based on token semantics.' }
  ]
};

export default function ResearchAnalyticsView({ api, onNavigate }) {
  return (
    <div style={{ padding: 28, background: colors.canvasParchment, color: colors.ink, height: '100%', overflowY: 'auto', fontFamily: typography.body.fontFamily }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 6px 0', color: colors.purpleBorder, display: 'flex', alignItems: 'center', gap: 8 }}><BarChart3 size={20} /> Multi-Campaign Meta-Learning Analytics</h1>
          <p style={{ margin: 0, color: colors.inkMuted48, fontSize: 13 }}>
            Meta-policy learning extracted across 150 historical research campaigns
          </p>
        </div>
        <div style={{ background: colors.surfacePearl, padding: '6px 14px', borderRadius: 8, border: `1px solid ${colors.hairline}`, fontSize: 12 }}>
          <span style={{ color: colors.inkMuted48 }}>Meta-Learning Policy: </span>
          <span style={{ color: colors.success, fontWeight: 600 }}>Active (v3.2)</span>
        </div>
      </div>

      {/* Executive Overview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <div style={{ background: colors.canvas, border: `1px solid ${colors.hairline}`, padding: 20, borderRadius: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, textTransform: 'uppercase' }}>Campaigns Analyzed</div>
          <div style={{ fontSize: 26, fontWeight: 800, color: colors.ink, marginTop: 4 }}>{MOCK_ANALYTICS.total_campaigns}</div>
        </div>
        <div style={{ background: colors.canvas, border: `1px solid ${colors.hairline}`, padding: 20, borderRadius: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, textTransform: 'uppercase' }}>Overall Success Rate</div>
          <div style={{ fontSize: 26, fontWeight: 800, color: colors.success, marginTop: 4 }}>{(MOCK_ANALYTICS.overall_success_rate * 100).toFixed(1)}%</div>
        </div>
        <div style={{ background: colors.canvas, border: `1px solid ${colors.hairline}`, padding: 20, borderRadius: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, textTransform: 'uppercase' }}>Total Compute Consumed</div>
          <div style={{ fontSize: 26, fontWeight: 800, color: colors.primary, marginTop: 4 }}>{MOCK_ANALYTICS.total_flops}</div>
        </div>
        <div style={{ background: colors.canvas, border: `1px solid ${colors.hairline}`, padding: 20, borderRadius: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: colors.inkMuted48, textTransform: 'uppercase' }}>Total Research Spend</div>
          <div style={{ fontSize: 26, fontWeight: 800, color: colors.warning, marginTop: 4 }}>${MOCK_ANALYTICS.total_usd.toFixed(2)}</div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 20 }}>
        
        {/* Left Column: Leaderboard & Best Sequences */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          
          {/* Algorithm Leaderboard */}
          <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 20 }}>
            <h3 style={{ fontSize: 15, color: colors.purpleBorder, margin: '0 0 14px 0', display: 'flex', alignItems: 'center', gap: 6 }}><Trophy size={15} /> Algorithm Performance Leaderboard</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
              <thead>
                <tr style={{ background: colors.surfacePearl, color: colors.inkMuted48, borderBottom: `1px solid ${colors.hairline}` }}>
                  <th style={{ padding: 10 }}>Algorithm</th>
                  <th style={{ padding: 10 }}>Success</th>
                  <th style={{ padding: 10 }}>Avg ΔBelief</th>
                  <th style={{ padding: 10 }}>Runtime</th>
                  <th style={{ padding: 10 }}>Gain / TFLOP</th>
                  <th style={{ padding: 10 }}>Gain / $1</th>
                </tr>
              </thead>
              <tbody>
                {MOCK_ANALYTICS.leaderboard.map((item, idx) => (
                  <tr key={idx} style={{ borderBottom: `1px solid ${colors.hairline}`, background: idx % 2 === 0 ? colors.canvas : colors.surfacePearl }}>
                    <td style={{ padding: 10, fontWeight: 700, color: colors.ink }}>{item.name}</td>
                    <td style={{ padding: 10, color: colors.success, fontWeight: 600 }}>{item.success}</td>
                    <td style={{ padding: 10, color: colors.primary, fontWeight: 600 }}>{item.avg_gain}</td>
                    <td style={{ padding: 10, color: colors.inkMuted48 }}>{item.runtime}</td>
                    <td style={{ padding: 10, color: colors.purpleBorder }}>{item.gain_tflop}</td>
                    <td style={{ padding: 10, color: colors.warning, fontWeight: 700 }}>{item.gain_usd}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Best Experiment Sequences */}
          <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 20 }}>
            <h3 style={{ fontSize: 15, color: colors.primary, margin: '0 0 14px 0', display: 'flex', alignItems: 'center', gap: 6 }}><Star size={15} /> Meta-Learned Optimal Experiment Sequences</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {MOCK_ANALYTICS.sequences.map(seq => (
                <div key={seq.rank} style={{ background: colors.surfacePearl, border: `1px solid ${colors.hairline}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>Rank #{seq.rank}: {seq.name}</span>
                    <span style={{ fontSize: 12, color: colors.success, fontWeight: 700 }}>Avg Posterior: {seq.posterior}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                    {seq.seq.map((step, sIdx) => (
                      <React.Fragment key={sIdx}>
                        <span style={{ background: colors.surfacePearl, border: `1px solid ${colors.primary}`, color: colors.primary, padding: '4px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600 }}>
                          {step}
                        </span>
                        {sIdx < seq.seq.length - 1 && <span style={{ color: colors.inkMuted48, display: 'inline-flex' }}><ArrowRight size={12} /></span>}
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>

        {/* Right Column: Failure Anti-Patterns & Compute Efficiency */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          
          {/* Failure Anti-Patterns Card */}
          <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 20 }}>
            <h3 style={{ fontSize: 15, color: colors.danger, margin: '0 0 12px 0', display: 'flex', alignItems: 'center', gap: 6 }}><TriangleAlert size={15} /> Learned Failure Anti-Patterns</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {MOCK_ANALYTICS.failures.map(f => (
                <div key={f.id} style={{ background: colors.surfacePearl, borderLeft: `3px solid ${colors.danger}`, borderRadius: '0 8px 8px 0', padding: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>{f.title}</span>
                    <span style={{ fontSize: 11, color: colors.danger, fontWeight: 700 }}>{f.count} Failures</span>
                  </div>
                  <div style={{ fontSize: 12, color: colors.ink, marginBottom: 6 }}>Root cause: {f.cause}</div>
                  <div style={{ fontSize: 11, color: colors.primary, background: colors.canvas, padding: 6, borderRadius: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Lightbulb size={12} style={{ flexShrink: 0 }} /> Remedy: {f.remedy}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Compute Efficiency Dashboard */}
          <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, padding: 20 }}>
            <h3 style={{ fontSize: 15, color: colors.warning, margin: '0 0 12px 0', display: 'flex', alignItems: 'center', gap: 6 }}><Zap size={15} /> Compute Efficiency Meter</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ background: colors.surfacePearl, padding: 12, borderRadius: 8, border: `1px solid ${colors.hairline}`, display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, color: colors.inkMuted48 }}>Confidence Gained per $1 USD</span>
                <span style={{ fontSize: 13, fontWeight: 700, color: colors.success }}>+4.25% / $1</span>
              </div>
              <div style={{ background: colors.surfacePearl, padding: 12, borderRadius: 8, border: `1px solid ${colors.hairline}`, display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, color: colors.inkMuted48 }}>Confidence Gained per TFLOP</span>
                <span style={{ fontSize: 13, fontWeight: 700, color: colors.primary }}>+0.21% / TFLOP</span>
              </div>
              <div style={{ background: colors.surfacePearl, padding: 12, borderRadius: 8, border: `1px solid ${colors.hairline}`, display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: 12, color: colors.inkMuted48 }}>Confidence Gained per GPU-sec</span>
                <span style={{ fontSize: 13, fontWeight: 700, color: colors.purpleBorder }}>+1.85% / sec</span>
              </div>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}

