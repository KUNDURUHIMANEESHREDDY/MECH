import React, { useState, useEffect, useCallback } from 'react';
import { ArrowRight, Check, Play } from 'lucide-react';
import { colors } from '../design/tokens/colors';

const MOCK_HISTORICAL_TRENDS = {
  'ioi': [95, 96, 96],
  'induction_heads': [88, 91, 94],
  'greater_than': [80, 85, 91],
  'logit_lens': [90, 90, 90],
  'sparse_autoencoders': [75, 80, 86],
};

function TierBadge({ tier }) {
  const TIER_COLORS = { Gold: colors.warning, Silver: colors.inkMuted48, Bronze: colors.warningText, 'Needs Investigation': colors.danger };
  return (
    <span style={{
      background: TIER_COLORS[tier] || colors.inkMuted48, color: tier === 'Gold' ? colors.ink : colors.onDark,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700}}>{tier}</span>
  );
}

function TrendSparkline({ values }) {
  if (!values || values.length === 0) return null;
  const isUp = values[values.length - 1] > values[0];
  const color = isUp ? colors.success : (values[values.length - 1] === values[0] ? colors.inkMuted48 : colors.danger);
  
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      {values.map((v, i) => (
        <React.Fragment key={i}>
<span style={{ fontSize: 11, color: i === values.length - 1 ? color : colors.inkMuted48 }}>{v}%</span>
          {i < values.length - 1 && <span style={{ fontSize: 10, color: colors.inkMuted80, display: 'inline-flex' }}><ArrowRight size={10} /></span>}
        </React.Fragment>
      ))}
    </div>
  );
}

export default function BenchmarkDashboard({ api }) {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadReports = useCallback(async () => {
    if (!api?.pythonCall) return;
    setLoading(true);
    try {
      const data = await api.pythonCall('api/v2/science/reproducibility_reports', {});
      if (data && data.reports) {
        setReports(data.reports);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    loadReports();
  }, [loadReports]);

  const runAllPipelines = async () => {
    // In a real app, this would trigger a background job on the backend
    alert('Starting full reproducibility suite. This will take several hours on a GPU.');
  };

  return (
    <div style={{ padding: 32, background: colors.canvasParchment, color: colors.ink, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
<h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 8px 0', color: colors.purpleBorder }}>Scientific Reproducibility</h1>
          <p style={{ margin: 0, color: colors.inkMuted48, fontSize: 14 }}>
            Continuous integration for mechanistic interpretability research.
          </p>
        </div>
        <button
          onClick={runAllPipelines}
          style={{
            background: `linear-gradient(135deg, ${colors.purple}, ${colors.primary})`, color: colors.onDark,
            border: 'none', borderRadius: 8, padding: '10px 24px', fontSize: 14,
            fontWeight: 600, cursor: 'pointer'}}
        >
          <><Play size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />Run Full Suite</>
        </button>
      </div>

      {loading ? (
<div style={{ color: colors.inkMuted48 }}>Loading benchmark reports...</div>
      ) : error ? (
        <div style={{ color: colors.danger }}>Error loading reports: {error}</div>
      ) : (
        <div style={{ background: colors.canvas, borderRadius: 12, border: `1px solid ${colors.hairline}`, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: colors.surfacePearl, borderBottom: `1px solid ${colors.hairline}` }}>
                <th style={{ padding: '16px 24px', color: colors.inkMuted48, fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Paper / Benchmark</th>
                <th style={{ padding: '16px 24px', color: colors.inkMuted48, fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Fidelity Tier</th>
                <th style={{ padding: '16px 24px', color: colors.inkMuted48, fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Trend History</th>
                <th style={{ padding: '16px 24px', color: colors.inkMuted48, fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Last Run</th>
                <th style={{ padding: '16px 24px', color: colors.inkMuted48, fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((report) => (
<tr key={report.report_id} style={{ borderBottom: `1px solid ${colors.hairline}` }}>
                  <td style={{ padding: '16px 24px' }}>
                    <div style={{ fontWeight: 600, color: colors.purpleBorder }}>
                      {report.paper_id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <div style={{ fontSize: 11, color: colors.inkMuted48, marginTop: 4 }}>
                      Model: {report.model_id}
                    </div>
                  </td>
                  <td style={{ padding: '16px 24px' }}>
                    <TierBadge tier={report.fidelity_tier} />
                  </td>
                  <td style={{ padding: '16px 24px' }}>
                    <TrendSparkline values={MOCK_HISTORICAL_TRENDS[report.paper_id]} />
                  </td>
<td style={{ padding: '16px 24px', fontSize: 12, color: colors.inkMuted48 }}>
                    {new Date(report.generated_at).toLocaleString()}
                  </td>
<td style={{ padding: '16px 24px', color: colors.success }}>
                    <><Check size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />Passing</>
                  </td>
                </tr>
              ))}
              {reports.length === 0 && (
                <tr>
                  <td colSpan={5} style={{ padding: '32px', textAlign: 'center', color: colors.inkMuted48 }}>
                    No reproducibility reports found. Run a pipeline to generate reports.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

