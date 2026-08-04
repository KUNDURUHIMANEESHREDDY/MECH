import React, { useState } from 'react';
import { Orbit } from 'lucide-react';
import { knowledgeExplorerEngine } from '../../services/knowledgeExplorerEngine.js';

export default function KnowledgeUniversePanel() {
  const [zoom, setZoom] = useState(1.0);
  const nodes = knowledgeExplorerEngine.getUniverseNodes();

  return (
    <div className="panel knowledge-universe-panel" data-testid="knowledge-universe-panel">
      <div className="panel-header">
        <h3><Orbit size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Knowledge Universe</h3>
        <div style={{ display: 'flex', gap: '4px' }}>
          <button className="btn" style={{ fontSize: '10px' }} onClick={() => setZoom((z) => Math.max(0.5, z - 0.2))}>Zoom -</button>
          <span style={{ fontSize: '11px', alignSelf: 'center', color: 'var(--text-muted)' }}>{(zoom * 100).toFixed(0)}%</span>
          <button className="btn" style={{ fontSize: '10px' }} onClick={() => setZoom((z) => Math.min(2.0, z + 0.2))}>Zoom +</button>
        </div>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ transform: `scale(${zoom})`, transformOrigin: 'top left', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {nodes.map((n) => (
            <div key={n.id} style={{ padding: '12px', background: 'var(--bg-elev-2)', borderRadius: '8px', border: '1px solid var(--border)' }}>
              <span className="badge" style={{ fontSize: '10px', background: 'var(--accent)', color: 'var(--bg)', padding: '2px 6px', borderRadius: '4px', fontWeight: 'bold' }}>
                {n.category}
              </span>
              <h4 style={{ margin: '4px 0', color: 'var(--text)', fontSize: '13px' }}>{n.label}</h4>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
