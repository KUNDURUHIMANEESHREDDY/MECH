import React from 'react';
import { FileText } from 'lucide-react';

export default function PublicationDashboardPanel() {
  const artifacts = [
    { title: 'Figure 1: IOI Causal Tracing Heatmap', type: 'SVG Figure', size: '240 KB' },
    { title: 'Table 1: Activation Patching Ablation Scores', type: 'LaTeX Table', size: '18 KB' },
    { title: 'Manuscript Draft: Mechanistic Reasoning Engine', type: 'Markdown Paper', size: '1.2 MB' },
    { title: 'Supplementary Data: 16k SAE Features Index', type: 'JSON Archive', size: '4.8 MB' }
  ];

  return (
    <div className="panel publication-dashboard-panel" data-testid="publication-dashboard-panel">
      <div className="panel-header">
        <h3><FileText size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Publication Dashboard</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {artifacts.map((a, i) => (
            <div key={i} style={{ padding: '10px 12px', background: 'var(--bg-elev-2)', borderRadius: '6px', border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong style={{ fontSize: '12px', color: 'var(--text)', display: 'block' }}>{a.title}</strong>
                <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{a.type} • {a.size}</span>
              </div>
              <button className="btn" style={{ fontSize: '11px', padding: '2px 8px' }}>Export</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
