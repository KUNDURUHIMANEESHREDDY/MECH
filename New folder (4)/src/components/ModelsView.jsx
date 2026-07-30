import React, { useState } from 'react';

export default function ModelsView({ api, onNavigate }) {
  const [selectedModel, setSelectedModel] = useState('gpt2');
  const [loading, setLoading]             = useState(false);
  const [modelInfo, setModelInfo]         = useState(null);
  const [error, setError]                 = useState(null);

  const MODELS = [
    { id: 'gpt2',        name: 'GPT-2 Small',           layers: 12, heads: 12, dModel: 768,  params: '124M',  status: 'Available' },
    { id: 'gpt2-medium', name: 'GPT-2 Medium',          layers: 24, heads: 16, dModel: 1024, params: '355M',  status: 'Available' },
    { id: 'pythia-160m', name: 'EleutherAI Pythia 160M', layers: 12, heads: 12, dModel: 768,  params: '160M',  status: 'Available' },
    { id: 'llama-3-8b',  name: 'Llama-3 8B (Quantized)', layers: 32, heads: 32, dModel: 4096, params: '8B',   status: 'Downloadable' },
  ];

  const handleLoad = async (modelId) => {
    setSelectedModel(modelId);
    if (modelId !== 'gpt2') {
      setModelInfo(null);
      setError('Only GPT-2 Small is currently wired to the live Python backend.');
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const r = await api.gpt2Load();
      setModelInfo(r);
      if (r.status === 'loaded' && onNavigate) {
        // brief delay then navigate to the GPT-2 live view
        setTimeout(() => onNavigate('gpt2'), 600);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="models-view" data-testid="models-view">
      <div className="section-header">
        <h2>Model Explorer</h2>
        <p>Inspect neural network architectures, layer configurations, and weights.</p>
      </div>

      {modelInfo?.status === 'loaded' && (
        <div className="card" style={{ marginBottom: 16, background: '#f0fdf4', border: '1px solid #bbf7d0' }}>
          <p style={{ margin: 0, fontWeight: 600, color: '#166534', fontSize: 13 }}>
            {modelInfo.model_name} loaded — {modelInfo.n_layers} layers · {modelInfo.n_heads} heads · d_model {modelInfo.d_model}
          </p>
          <p style={{ margin: '4px 0 0', fontSize: 12, color: '#166534' }}>
            Navigating to GPT-2 Live view…
          </p>
        </div>
      )}
      {error && (
        <div className="card" style={{ marginBottom: 16, background: '#fef2f2', border: '1px solid #fecaca' }}>
          <p style={{ margin: 0, color: '#991b1b', fontSize: 13 }}>{error}</p>
        </div>
      )}

      <div className="models-grid">
        {MODELS.map((m) => (
          <div
            key={m.id}
            className={`model-card ${selectedModel === m.id ? 'active' : ''}`}
            onClick={() => setSelectedModel(m.id)}
          >
            <div className="model-card-header">
              <h3>{m.name}</h3>
              <span className={`status-badge ${m.status.toLowerCase()}`}>{m.status}</span>
            </div>
            <div className="model-specs">
              <div><span>Layers:</span>  <strong>{m.layers}</strong></div>
              <div><span>Heads:</span>   <strong>{m.heads}</strong></div>
              <div><span>d_model:</span> <strong>{m.dModel}</strong></div>
              <div><span>Params:</span>  <strong>{m.params}</strong></div>
            </div>
            <button
              className="btn btn-primary btn-sm"
              style={{ marginTop: 12, width: '100%' }}
              disabled={loading && selectedModel === m.id}
              onClick={(e) => { e.stopPropagation(); handleLoad(m.id); }}
            >
              {loading && selectedModel === m.id
                ? 'Loading…'
                : selectedModel === m.id && modelInfo?.status === 'loaded'
                  ? 'Loaded ✓'
                  : 'Load Model'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
