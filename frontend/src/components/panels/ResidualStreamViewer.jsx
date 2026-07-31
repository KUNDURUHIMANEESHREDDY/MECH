import React from 'react';
import { Zap } from 'lucide-react';
import { selectionManager } from '../../utils/selectionManager';

export default function ResidualStreamViewer() {
  const LAYERS = Array.from({ length: 12 }, (_, i) => ({
    layer: i + 1,
    norm: (12.4 + i * 1.8).toFixed(1),
    cosine: (0.95 - i * 0.05).toFixed(2),
    hasPatch: i === 5 || i === 8,
  }));

  return (
    <div className="panel-content residual-stream-panel" data-testid="residual-stream-panel">
      <h4>Residual Stream Vector Evolution</h4>
      <p className="hint">Layer-by-layer residual stream vector norms, cosine similarities, and patch markers.</p>

      <div className="residual-bars-container" style={{ display: 'flex', gap: '8px', alignItems: 'flex-end', height: '120px', padding: '10px 0' }}>
        {LAYERS.map((l) => (
          <div
            key={l.layer}
            className="residual-bar-col"
            onClick={() => selectionManager.setLayer(l.layer)}
            style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'flex-end' }}
          >
            {l.hasPatch && <span className="patch-marker" title="Patch applied"><Zap size={10} /></span>}
            <div
              className="bar-fill"
              style={{
                height: `${(Number(l.norm) / 32) * 100}%`,
                width: '100%',
                background: l.hasPatch ? 'var(--warning)' : 'var(--accent-strong)',
                borderRadius: '4px 4px 0 0',
              }}
            />
            <span className="bar-label" style={{ fontSize: '10px', marginTop: '4px' }}>L{l.layer}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
