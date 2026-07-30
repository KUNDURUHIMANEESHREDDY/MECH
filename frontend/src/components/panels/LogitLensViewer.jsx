import React, { useState, useEffect } from 'react';
import { visualizationService } from '../../services/visualizationService';

export default function LogitLensViewer() {
  const [method, setMethod] = useState('logit_lens');
  const [lensData, setLensData] = useState(null);

  useEffect(() => {
    visualizationService.fetchLogitLens('The capital of France is', 11, method)
      .then((res) => setLensData(res))
      .catch(() => undefined);
  }, [method]);

  return (
    <div className="panel-content logit-lens-panel" data-testid="logit-lens-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <h4>Projection Lens Viewer ({lensData ? lensData.method : 'LogitLens'})</h4>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button className={`btn btn-sm ${method === 'logit_lens' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMethod('logit_lens')}>
            Logit Lens
          </button>
          <button className={`btn btn-sm ${method === 'tuned_lens' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMethod('tuned_lens')}>
            Tuned Lens
          </button>
        </div>
      </div>

      {lensData ? (
        <div className="lens-projections-card" style={{ background: 'var(--bg)', padding: '12px', borderRadius: '6px' }}>
          <div style={{ marginBottom: '8px', fontSize: '12px' }}>
            Prompt: <strong>"{lensData.prompt}"</strong> | Top Prediction: <strong>{lensData.top_token}</strong> (Entropy: {lensData.entropy})
          </div>
          <div className="tokens-table" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {(lensData.top_k_tokens || []).map((t, idx) => (
              <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', padding: '4px 8px', background: 'var(--bg-elev)', borderRadius: '4px' }}>
                <span style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>{t.token}</span>
                <span>Logit: {t.logit}</span>
                <span style={{ color: 'var(--success)' }}>{(t.probability * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <p className="hint">Extracting intermediate layer unembedding projections...</p>
      )}
    </div>
  );
}
