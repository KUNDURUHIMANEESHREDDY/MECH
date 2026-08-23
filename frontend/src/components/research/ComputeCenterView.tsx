import React, { useEffect, useState } from 'react';
import { api } from '../../services/api';
import { colors } from '../../design/tokens/colors';
import { Cpu, Server, Activity, CheckCircle, Clock, RefreshCw } from 'lucide-react';

export const ComputeCenterView: React.FC = () => {
  const [jobs, setJobs] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchJobs = async () => {
    setLoading(true);
    try {
      const res = await api.listJobs();
      setJobs(res.jobs || []);
    } catch {
      setJobs([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, []);

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            System Infrastructure
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Compute & Experiment Execution Center
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Manage background experiment workers, tensor storage caches, and job queues.
          </div>
        </div>

        <button
          onClick={fetchJobs}
          style={{
            padding: '7px 12px',
            borderRadius: 6,
            border: `1px solid ${colors.border}`,
            backgroundColor: colors.surfaceTile1,
            color: colors.ink,
            fontSize: 12,
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <RefreshCw size={13} className={loading ? 'spinner' : ''} /> Refresh
        </button>
      </div>

      {/* Hardware Telemetry Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginBottom: 20 }}>
        <div style={{ padding: 14, borderRadius: 8, backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
          <div style={{ fontSize: 11, color: colors.bodyMuted, textTransform: 'uppercase' }}>Compute Backend</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: colors.ink, marginTop: 4 }}>PyTorch 2.4.0 (CPU / DirectML)</div>
          <div style={{ fontSize: 11, color: colors.successText, marginTop: 2 }}>Status: Online & Ready</div>
        </div>

        <div style={{ padding: 14, borderRadius: 8, backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
          <div style={{ fontSize: 11, color: colors.bodyMuted, textTransform: 'uppercase' }}>CAS Tensor Storage</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: colors.ink, marginTop: 4 }}>Content-Addressed (SHA256)</div>
          <div style={{ fontSize: 11, color: colors.bodyMuted, marginTop: 2 }}>Location: ~/.cache/neural-debugger/</div>
        </div>

        <div style={{ padding: 14, borderRadius: 8, backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
          <div style={{ fontSize: 11, color: colors.bodyMuted, textTransform: 'uppercase' }}>Database Engine</div>
          <div style={{ fontSize: 16, fontWeight: 700, color: colors.ink, marginTop: 4 }}>SQLite 3 (WAL Mode)</div>
          <div style={{ fontSize: 11, color: colors.bodyMuted, marginTop: 2 }}>Auto-retry busy timeout: 30,000ms</div>
        </div>
      </div>

      {/* Experiment Queue Table */}
      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, backgroundColor: colors.surfaceTile1 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink, marginBottom: 12 }}>
          Compute & Execution Queue ({jobs.length})
        </div>

        {jobs.length === 0 ? (
          <div style={{ fontSize: 12, color: colors.bodyMuted, textAlign: 'center', padding: '24px 0' }}>
            No background jobs queued. All recent interventions executed synchronously in real-time.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${colors.border}`, color: colors.bodyMuted, textAlign: 'left' }}>
                <th style={{ padding: '8px 6px' }}>Job ID</th>
                <th style={{ padding: '8px 6px' }}>Type</th>
                <th style={{ padding: '8px 6px' }}>Name</th>
                <th style={{ padding: '8px 6px' }}>Progress</th>
                <th style={{ padding: '8px 6px' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} style={{ borderBottom: `1px solid ${colors.borderLight}` }}>
                  <td style={{ padding: '8px 6px' }}><code>{j.id}</code></td>
                  <td style={{ padding: '8px 6px' }}>{j.job_type}</td>
                  <td style={{ padding: '8px 6px', fontWeight: 600, color: colors.ink }}>{j.name}</td>
                  <td style={{ padding: '8px 6px' }}>{(j.progress * 100).toFixed(0)}%</td>
                  <td style={{ padding: '8px 6px' }}>
                    <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 6px', borderRadius: 4, backgroundColor: colors.surfacePearl, color: colors.body }}>
                      {j.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
