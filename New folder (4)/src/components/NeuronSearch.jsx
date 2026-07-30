import React, { useState } from 'react';

export default function NeuronSearch({ onSelectNeuron }) {
  const [query, setQuery] = useState('');
  const [layerFilter, setLayerFilter] = useState('all');
  const [minActivation, setMinActivation] = useState(0.0);

  const NEURONS = [
    { layer: 0, index: 14, feature: 'First Token Invariant', act: 1.42 },
    { layer: 3, index: 104, feature: 'Punctuation Probe', act: 0.88 },
    { layer: 8, index: 402, feature: 'Induction Head Key', act: 2.41 },
    { layer: 9, index: 9, feature: 'Indirect Object Attn', act: 3.12 },
    { layer: 11, index: 118, feature: 'Previous Token Copying', act: 1.89 }
  ];

  const filtered = NEURONS.filter((n) => {
    const matchesQuery = query === '' ||
      `L${n.layer}_N${n.index}`.toLowerCase().includes(query.toLowerCase()) ||
      n.feature.toLowerCase().includes(query.toLowerCase());
    const matchesLayer = layerFilter === 'all' || n.layer === Number(layerFilter);
    const matchesAct = n.act >= minActivation;
    return matchesQuery && matchesLayer && matchesAct;
  });

  return (
    <div className="neuron-search-container" data-testid="neuron-search">
      <div className="neuron-search-controls" style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
        <input
          type="text"
          className="input-text"
          placeholder="Search neuron ID or feature (e.g. L8_N402 or Induction)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ flex: 2 }}
        />
        <select
          value={layerFilter}
          onChange={(e) => setLayerFilter(e.target.value)}
          style={{ flex: 1 }}
        >
          <option value="all">All Layers</option>
          <option value="0">Layer 0</option>
          <option value="3">Layer 3</option>
          <option value="8">Layer 8</option>
          <option value="9">Layer 9</option>
          <option value="11">Layer 11</option>
        </select>
      </div>

      <div className="neuron-results-list">
        {filtered.length === 0 ? (
          <p className="hint">No neurons matching search filter.</p>
        ) : (
          filtered.map((n) => (
            <div
              key={`L${n.layer}_N${n.index}`}
              className="neuron-search-item"
              onClick={() => onSelectNeuron && onSelectNeuron(n)}
            >
              <div className="neuron-id-tag">L{n.layer}_N{n.index}</div>
              <div className="neuron-feature-name">{n.feature}</div>
              <div className="neuron-act-val">+{n.act}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
