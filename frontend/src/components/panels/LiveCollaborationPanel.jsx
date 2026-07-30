import React from 'react';
import { sharedWorkspaceState } from '../../utils/sharedWorkspaceState.js';

export default function LiveCollaborationPanel() {
  const { activeParticipants } = sharedWorkspaceState.getState();

  return (
    <div className="panel live-collaboration-panel" data-testid="live-collaboration-panel">
      <div className="panel-header">
        <h3>👥 Live Collaboration View</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {activeParticipants.map((p) => (
            <div key={p.id} style={{ padding: '10px 12px', background: '#1e293b', borderRadius: '6px', border: '1px solid #334155', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong style={{ fontSize: '12px', color: '#f8fafc', display: 'block' }}>{p.name}</strong>
                <span style={{ fontSize: '10px', color: '#94a3b8' }}>{p.role}</span>
              </div>
              <span className="badge" style={{ fontSize: '10px', background: p.status === 'Active' ? '#10b981' : '#64748b', color: '#fff', padding: '2px 6px', borderRadius: '4px' }}>
                {p.status}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
