import React, { useEffect, useState } from 'react';
import { LogitLensAll } from '../types';
import { fetchLogitLensAll } from '../services/inferenceService';

interface PredictionPanelProps {
  prompt: string;
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({ prompt }) => {
  const [lens, setLens] = useState<LogitLensAll | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!prompt.trim()) {
      setLens(null);
      return;
    }
    let live = true;
    setLoading(true);
    setError(null);
    fetchLogitLensAll(prompt)
      .then(data => {
        if (live) {
          setLens(data);
          setLoading(false);
        }
      })
      .catch((e: unknown) => {
        if (live) {
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      });
    return () => {
      live = false;
    };
  }, [prompt]);

  const cardBg = 'var(--bg-elev-2)';
  const border = 'var(--border)';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12 }}>
      <div style={{ background: cardBg, padding: 10, borderRadius: 6, border: `1px solid ${border}` }}>
        <div style={{ fontWeight: 700, fontSize: 13, color: 'var(--purple)' }}>Prediction Inspector & Logit Lens</div>
        <div style={{ color: 'var(--text-muted)', marginTop: 2 }}>
          Live unembedding projection per layer — no trained translators involved
        </div>
      </div>

      {loading && <div style={{ color: 'var(--text-muted)' }}>Projecting {lens?.layers.length ?? 12} layers…</div>}
      {error && <div style={{ color: 'var(--danger)' }}>{error}</div>}

      {!loading && !error && lens && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {lens.layers.map(l => {
            const top = l.top_k_tokens[0];
            return (
              <div
                key={l.layer}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  background: cardBg,
                  padding: '8px 12px',
                  borderRadius: 6,
                  border: `1px solid ${border}`,
                }}
              >
                <div style={{ fontWeight: 600 }}>Layer {l.layer} Logit Lens</div>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                  <span style={{ background: 'var(--purple)', color: 'var(--bg)', padding: '2px 8px', borderRadius: 4, fontWeight: 700 }}>
                    &quot;{l.top_token}&quot; ({((top?.prob ?? 0) * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
