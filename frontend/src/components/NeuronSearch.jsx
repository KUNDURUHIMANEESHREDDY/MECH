import React, { useState, useEffect } from 'react';

export default function NeuronSearch({ api, onSelectNeuron, selectedLayer }) {
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(0);
  const [sortBy, setSortBy] = useState('activation');
  const [order, setOrder] = useState('desc');
  const [neurons, setNeurons] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const pageSize = 50;

  const layer = selectedLayer ?? 0;

  const loadNeurons = async (overridePage, overrideSort, overrideOrder) => {
    const p = overridePage !== undefined ? overridePage : page;
    const s = overrideSort !== undefined ? overrideSort : sortBy;
    const o = overrideOrder !== undefined ? overrideOrder : order;
    setLoading(true);
    setError(null);
    try {
      const res = await api.gpt2Neurons(layer, 'mlp', p, pageSize, s, o);
      if (res.status === 'error') throw new Error(res.error);
      setNeurons(res.neurons || []);
      setTotal(res.total_neurons || 0);
      setPage(p);
      setSortBy(s);
      setOrder(o);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Load on mount and when layer/sort/order changes
  useEffect(() => {
    loadNeurons(0, 'activation', 'desc');
  }, [layer]);

  const filtered = neurons.filter((n) => {
    const q = query.toLowerCase();
    return q === '' ||
      n.label.toLowerCase().includes(q) ||
      n.neuron_index.toString().includes(q);
  });

  // Render a dynamic grid of neurons with real activation values
  return (
    <div className="neuron-search-container" data-testid="neuron-search">
      <div className="neuron-search-controls" style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
        <input
          type="text"
          className="input-text"
          placeholder="Search neuron ID or label (e.g. L8_N402)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ flex: 2 }}
        />
        <select
          value={sortBy}
          onChange={(e) => loadNeurons(0, e.target.value, order)}
          style={{ flex: 1 }}
        >
          <option value="activation">Activation</option>
          <option value="in_weight_l2">Input Weight L2</option>
          <option value="out_weight_l2">Output Weight L2</option>
          <option value="bias">Bias</option>
          <option value="index">Index</option>
        </select>
        <select
          value={order}
          onChange={(e) => loadNeurons(0, sortBy, e.target.value)}
          style={{ flex: 1 }}
        >
          <option value="desc">Desc</option>
          <option value="asc">Asc</option>
        </select>
      </div>

      {error && (
        <p style={{ color: '#dc2626', fontSize: 12, marginTop: 8 }}>{error}</p>
      )}

      {loading ? (
        <div className="skeleton-container">
          <div className="skeleton-box" style={{ height: 20, width: '80%' }} />
          <div className="skeleton-box" style={{ height: 20, width: '90%' }} />
          <div className="skeleton-box" style={{ height: 20, width: '70%' }} />
        </div>
      ) : (
        <div className="neuron-results-list">
          {filtered.length === 0 ? (
            <p className="hint">
              {total > 0 ? 'No neurons matching search.' : 'Run a prompt first to populate neuron activations.'}
            </p>
          ) : (
            filtered.map((n) => (
              <div
                key={n.label}
                className="neuron-search-item"
                onClick={() => onSelectNeuron && onSelectNeuron(n)}
                style={{
                  cursor: 'pointer',
                  borderLeft: n.activation > 0.5 ? '3px solid #3b82f6' : '3px solid transparent',
                }}
              >
                <div className="neuron-id-tag">{n.label}</div>
                <div className="neuron-feature-name">
                  in_w: {n.in_weight_l2 !== null ? n.in_weight_l2.toFixed(3) : '—'} ·
                  out_w: {n.out_weight_l2 !== null ? n.out_weight_l2.toFixed(3) : '—'}
                </div>
                <div className="neuron-act-val">
                  {n.activation !== null ? n.activation.toFixed(3) : '—'}
                </div>
              </div>
            ))
          )}

          {total > pageSize && (
            <div style={{ display: 'flex', gap: 8, marginTop: 8, justifyContent: 'center' }}>
              <button
                className="btn btn-secondary btn-xs"
                onClick={() => loadNeurons(page - 1, sortBy, order)}
                disabled={page === 0}
              >
                Prev
              </button>
              <span style={{ fontSize: 11, color: '#888', lineHeight: '24px' }}>
                Page {page + 1} · {total} total neurons
              </span>
              <button
                className="btn btn-secondary btn-xs"
                onClick={() => loadNeurons(page + 1, sortBy, order)}
                disabled={!filtered.length || (page + 1) * pageSize >= total}
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
