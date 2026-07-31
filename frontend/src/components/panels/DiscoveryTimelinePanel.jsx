import React from 'react';
import { Hourglass } from 'lucide-react';

export default function DiscoveryTimelinePanel() {
  const milestones = [
    { date: 'Month 1', title: 'IOI Induction Head Identified' },
    { date: 'Month 2', title: 'SAE Polysemantic Feature Cataloged' },
    { date: 'Month 3', title: 'Cross-Family Gemma Alignment Verified' }
  ];

  return (
    <div className="panel discovery-timeline-panel" data-testid="discovery-timeline-panel">
      <div className="panel-header">
        <h3><Hourglass size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Discovery Evolution Timeline</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {milestones.map((m, i) => (
            <div key={i} style={{ padding: '10px', background: '#1e293b', borderRadius: '6px', borderLeft: '4px solid #a855f7' }}>
              <span style={{ fontSize: '10px', color: '#a855f7', fontWeight: 'bold' }}>{m.date}</span>
              <h5 style={{ margin: '4px 0 0 0', color: '#f8fafc' }}>{m.title}</h5>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
