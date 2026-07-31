import React, { useState, useEffect, useCallback } from 'react';
import { ArrowRight, Check, Play } from 'lucide-react';

const MOCK_HISTORICAL_TRENDS = {
  'ioi': [95, 96, 96],
  'induction_heads': [88, 91, 94],
  'greater_than': [80, 85, 91],
  'logit_lens': [90, 90, 90],
  'sparse_autoencoders': [75, 80, 86],
};

function TierBadge({ tier }) {
  const TIER_COLORS = { Gold: '#f5c518', Silver: '#aaa', Bronze: '#cd7f32', 'Needs Investigation': '#e55' };
  return (
    <span style={{
      background: TIER_COLORS[tier] || '#555', color: tier === 'Gold' ? '#111' : '#fff',
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700}}>{tier}</span>
  );
}

function TrendSparkline({ values }) {
  if (!values || values.length === 0) return null;
  const isUp = values[values.length - 1] > values[0];
  const color = isUp ? '#5cd4c4' : (values[values.length - 1] === values[0] ? '#888' : '#e55');
  
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      {values.map((v, i) => (
        <React.Fragment key={i}>
          <span style={{ fontSize: 11, color: i === values.length - 1 ? color : '#666' }}>{v}%</span>
          {i < values.length - 1 && <span style={{ fontSize: 10, color: '#444', display: 'inline-flex' }}><ArrowRight size={10} /></span>}
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
    <div style={{ padding: 32, background: '#0b0b1a', color: '#e0e0ff', height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 8px 0', color: '#d0c0ff' }}>Scientific Reproducibility</h1>
          <p style={{ margin: 0, color: '#888', fontSize: 14 }}>
            Continuous integration for mechanistic interpretability research.
          </p>
        </div>
        <button
          onClick={runAllPipelines}
          style={{
            background: 'linear-gradient(135deg, #7c6af7, #5cd4c4)', color: '#fff',
            border: 'none', borderRadius: 8, padding: '10px 24px', fontSize: 14,
            fontWeight: 600, cursor: 'pointer'}}
        >
          <><Play size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />Run Full Suite</>
        </button>
      </div>

      {loading ? (
        <div style={{ color: '#888' }}>Loading benchmark reports...</div>
      ) : error ? (
        <div style={{ color: '#e55' }}>Error loading reports: {error}</div>
      ) : (
        <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: '#1a1a2e', borderBottom: '1px solid #2a2a4a' }}>
                <th style={{ padding: '16px 24px', color: '#888', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Paper / Benchmark</th>
                <th style={{ padding: '16px 24px', color: '#888', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Fidelity Tier</th>
                <th style={{ padding: '16px 24px', color: '#888', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Trend History</th>
                <th style={{ padding: '16px 24px', color: '#888', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Last Run</th>
                <th style={{ padding: '16px 24px', color: '#888', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((report) => (
                <tr key={report.report_id} style={{ borderBottom: '1px solid #1e1e3a' }}>
                  <td style={{ padding: '16px 24px' }}>
                    <div style={{ fontWeight: 600, color: '#c0b0ff' }}>
                      {report.paper_id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>
                      Model: {report.model_id}
                    </div>
                  </td>
                  <td style={{ padding: '16px 24px' }}>
                    <TierBadge tier={report.fidelity_tier} />
                  </td>
                  <td style={{ padding: '16px 24px' }}>
                    <TrendSparkline values={MOCK_HISTORICAL_TRENDS[report.paper_id]} />
                  </td>
                  <td style={{ padding: '16px 24px', fontSize: 12, color: '#888' }}>
                    {new Date(report.generated_at).toLocaleString()}
                  </td>
                  <td style={{ padding: '16px 24px', color: '#5cd4c4' }}>
                    <><Check size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />Passing</>
                  </td>
                </tr>
              ))}
              {reports.length === 0 && (
                <tr>
                  <td colSpan={5} style={{ padding: '32px', textAlign: 'center', color: '#666' }}>
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

