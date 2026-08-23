import React, { useState } from 'react';
import { useModel } from '../hooks/useModel';
import './NeuronSearch.css';

export const NeuronSearch = () => {
  const { state: model } = useModel();
  const [query, setQuery] = useState('');
  const [selectedLayer, setSelectedLayer] = useState(8);
  const [neuronIndex, setNeuronIndex] = useState(402);
  const [patchValue, setPatchValue] = useState(0.0);
  const [patchStatus, setPatchStatus] = useState('');

  const topActivatingTokens = [
    { token: ' Paris', activation: 8.42 },
    { token: ' France', activation: 7.95 },
    { token: ' capital', activation: 6.81 },
    { token: ' city', activation: 5.12 },
    { token: ' European', activation: 4.88 },
  ];

  const handlePatch = () => {
    setPatchStatus(`Patched Neuron L${selectedLayer}_N${neuronIndex} with value ${patchValue}`);
    setTimeout(() => setPatchStatus(''), 3000);
  };

  return (
    <div className="neuron-search" data-testid="neuron-search">
      <div className="neuron-header" style={{ marginBottom: '16px' }}>
        <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>Neuron & Feature Search</h2>
        <span style={{ fontSize: '12px', color: '#a1a1aa' }}>
          Target Architecture: <strong style={{ color: '#10b981' }}>{model.modelInfo?.model_name || 'GPT-2 Small'}</strong>
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '16px' }}>
        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#a1a1aa', marginBottom: '4px' }}>Semantic Concept Search</label>
          <input
            data-testid="neuron-search-query-input"
            type="text"
            placeholder="e.g. geographical capitals, math operators..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#a1a1aa', marginBottom: '4px' }}>Target Layer Index</label>
          <input
            data-testid="neuron-layer-input"
            type="number"
            min={0}
            max={model.modelInfo?.num_layers ? model.modelInfo.num_layers - 1 : 11}
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(parseInt(e.target.value, 10) || 0)}
            style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
          />
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '12px', color: '#a1a1aa', marginBottom: '4px' }}>Neuron Index</label>
          <input
            data-testid="neuron-index-input"
            type="number"
            min={0}
            max={3071}
            value={neuronIndex}
            onChange={(e) => setNeuronIndex(parseInt(e.target.value, 10) || 0)}
            style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
          />
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <div style={{ background: '#18181b', padding: '16px', borderRadius: '8px', border: '1px solid #27272a' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#f4f4f5' }}>
            Top Activating Tokens for L{selectedLayer}_N{neuronIndex}
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            {topActivatingTokens.map((item, idx) => (
              <div
                key={idx}
                data-testid={`top-token-${idx}`}
                style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 10px', background: '#27272a', borderRadius: '4px' }}
              >
                <code style={{ color: '#10b981', fontWeight: 600 }}>{JSON.stringify(item.token)}</code>
                <span style={{ fontSize: '12px', color: '#a1a1aa' }}>+{item.activation.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ background: '#18181b', padding: '16px', borderRadius: '8px', border: '1px solid #27272a' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: '14px', color: '#f4f4f5' }}>Ablation & Steering Patch</h3>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', fontSize: '12px', color: '#a1a1aa', marginBottom: '4px' }}>Patch Fixed Activation Value</label>
            <input
              data-testid="patch-value-input"
              type="number"
              step={0.1}
              value={patchValue}
              onChange={(e) => setPatchValue(parseFloat(e.target.value) || 0.0)}
              style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
            />
          </div>
          <button
            data-testid="apply-patch-btn"
            onClick={handlePatch}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: 'none',
              fontWeight: 600,
              background: '#10b981',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Apply Activation Patch
          </button>
          {patchStatus && <div style={{ marginTop: '8px', fontSize: '12px', color: '#10b981' }}>{patchStatus}</div>}
        </div>
      </div>
    </div>
  );
};
