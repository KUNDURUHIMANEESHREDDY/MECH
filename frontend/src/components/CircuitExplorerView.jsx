import React, { useEffect, useState } from 'react';
import { colors } from '../design/tokens/colors';

const API_BASE = 'http://localhost:8000/api';

async function getJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path}: ${res.status}`);
  return res.json();
}

export default function CircuitExplorerView({ api }) {
  const [circuits, setCircuits] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let live = true;
    getJSON('/circuits')
      .then(data => {
        if (!live) return;
        const list = Array.isArray(data) ? data : data.circuits ?? [];
        setCircuits(list);
        if (list.length > 0) setSelectedId(list[0].circuit_id);
        setLoading(false);
      })
      .catch(e => {
        if (live) {
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      });
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let live = true;
    setDetail(null);
    getJSON(`/circuits/${encodeURIComponent(selectedId)}`)
      .then(data => {
        if (live) setDetail(data);
      })
      .catch(() => {
        if (live) setDetail(null);
      });
    return () => {
      live = false;
    };
  }, [selectedId]);

  const nodes = detail?.nodes ?? [];
  const heads = nodes.filter(n => n.node_type === 'attention_head');
  const others = nodes.filter(n => n.node_type !== 'attention_head');
  const score = detail ? detail.faithfulness ?? null : null;

  return (
    <div className="explorer-page">
      <div className="explorer-header">
        <h1>Circuit Explorer</h1>
        <p className="hint">Hierarchical exploration of mechanistic circuits.</p>
      </div>

      {loading && <p className="hint">Loading circuits from the backend…</p>}
      {error && (
        <p className="hint" style={{ color: 'var(--danger, #ef4444)' }}>
          Cannot reach the circuit registry: {error}. Start the backend first.
        </p>
      )}

      {!loading && !error && circuits.length === 0 && (
        <p className="hint">No circuits registered.</p>
      )}

      {!loading && !error && circuits.length > 0 && (
        <div className="explorer-split">
          <div className="explorer-panel">
            <h2>Circuit Selection</h2>

            <select
              value={selectedId}
              onChange={e => setSelectedId(e.target.value)}
              className="input-text"
              style={{ marginBottom: 24 }}
            >
              {circuits.map(c => (
                <option key={c.circuit_id} value={c.circuit_id}>
                  {c.name ?? c.circuit_id}
                </option>
              ))}
            </select>

            {detail && (
              <>
                <div className="evidence-block" style={{ borderLeftColor: colors.primary }}>
                  <div className="evidence-label">Evidence</div>
                  <p className="evidence-text">{detail.description ?? '—'}</p>
                </div>

                <div className="evidence-block" style={{ borderLeftColor: colors.warning }}>
                  <div className="evidence-label">Faithfulness</div>
                  <div className="confidence-bar">
                    <div className="confidence-track">
                      <div
                        className="confidence-fill"
                        style={{
                          width: `${Math.round((score ?? 0) * 100)}%`,
                          background: colors.warning,
                        }}
                      />
                    </div>
                    <span className="confidence-value">
                      {score === null ? '—' : `${Math.round(score * 100)}%`}
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="explorer-panel" style={{ flex: 2 }}>
            <h2>Members & Components</h2>

            {!detail && <p className="hint">Select a circuit to inspect its members.</p>}

            {detail && (
              <div className="component-list">
                <div className="component-card">
                  <div className="component-header">
                    <h3 style={{ color: colors.bodyMuted }}>Attention Heads</h3>
                  </div>
                  <div className="component-tags">
                    {heads.length === 0 && <span className="hint">None listed</span>}
                    {heads.map(h => (
                      <span className="chip" key={h.node_id} title={h.description ?? ''}>
                        {h.label}
                        {h.importance_score !== undefined ? ` (${h.importance_score})` : ''}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="component-card">
                  <h3 style={{ color: colors.ink, margin: '0 0 12px 0' }}>Other Components</h3>
                  <div className="component-tags">
                    {others.length === 0 && <span className="hint">None listed</span>}
                    {others.map(n => (
                      <span className="chip" key={n.node_id} title={n.description ?? ''}>
                        {n.label}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
