import React from 'react';
import { Users } from 'lucide-react';
import { sharedWorkspaceState } from '../../utils/sharedWorkspaceState.js';

export default function LiveCollaborationPanel() {
  const { activeParticipants } = sharedWorkspaceState.getState();

  return (
    <div className="panel live-collaboration-panel" data-testid="live-collaboration-panel">
      <div className="panel-header">
        <h3><Users size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Live Collaboration View</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {activeParticipants.map((p) => (
            <div key={p.id} style={{ padding: '10px 12px', background: 'var(--bg-elev-2)', borderRadius: '6px', border: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <strong style={{ fontSize: '12px', color: 'var(--text)', display: 'block' }}>{p.name}</strong>
                <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{p.role}</span>
              </div>
              <span className="badge" style={{ fontSize: '10px', background: p.status === 'Active' ? 'var(--success)' : 'var(--text-muted)', color: 'var(--bg)', padding: '2px 6px', borderRadius: '4px' }}>
                {p.status}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
