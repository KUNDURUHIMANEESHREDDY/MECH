import React, { useState } from 'react';
import { useModel } from '../hooks/useModel';
import './TransformerExplorer.css';

export const TransformerExplorer: React.FC = () => {
  const { state: model } = useModel();
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [activeComponent, setActiveComponent] = useState<'attn' | 'mlp' | 'ln'>('attn');
  const [prompt, setPrompt] = useState('When Mary and John went to the store, John gave a drink to');

  const totalLayers = model.modelInfo?.num_layers || 12;
  const numHeads = model.modelInfo?.num_heads || 12;
  const hiddenSize = model.modelInfo?.hidden_size || 768;

  return (
    <div className="transformer-explorer" data-testid="transformer-explorer">
      <div className="explorer-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>Transformer Architecture Explorer</h2>
          <span style={{ fontSize: '12px', color: '#a1a1aa' }}>
            Model: <strong style={{ color: '#10b981' }}>{model.modelInfo?.model_name || 'GPT-2 Small'}</strong> ({totalLayers} Layers, {numHeads} Heads, d_model={hiddenSize})
          </span>
        </div>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', fontSize: '12px', color: '#a1a1aa', marginBottom: '4px' }}>Input Inspection Prompt</label>
        <input
          data-testid="explorer-prompt-input"
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          style={{ width: '100%', boxSizing: 'border-box', padding: '8px 12px', background: '#27272a', border: '1px solid #3f3f46', borderRadius: '6px', color: '#fff' }}
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: '16px' }}>
        <div style={{ background: '#18181b', padding: '12px', borderRadius: '8px', border: '1px solid #27272a' }}>
          <span style={{ fontSize: '12px', fontWeight: 600, color: '#a1a1aa', display: 'block', marginBottom: '8px' }}>Layers (0 to {totalLayers - 1})</span>
          <div style={{ maxHeight: '300px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {Array.from({ length: totalLayers }).map((_, idx) => (
              <button
                key={idx}
                data-testid={`layer-select-btn-${idx}`}
                onClick={() => setSelectedLayer(idx)}
                style={{
                  padding: '6px 10px',
                  borderRadius: '4px',
                  border: 'none',
                  textAlign: 'left',
                  cursor: 'pointer',
                  background: selectedLayer === idx ? '#10b981' : '#27272a',
                  color: selectedLayer === idx ? '#fff' : '#d4d4d8',
                  fontSize: '13px',
                }}
              >
                Layer {idx}
              </button>
            ))}
          </div>
        </div>

        <div style={{ background: '#18181b', padding: '16px', borderRadius: '8px', border: '1px solid #27272a' }}>
          <div style={{ display: 'flex', gap: '8px', marginBottom: '14px' }}>
            <button
              data-testid="tab-attn"
              onClick={() => setActiveComponent('attn')}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                background: activeComponent === 'attn' ? '#3f3f46' : '#27272a',
                color: activeComponent === 'attn' ? '#10b981' : '#a1a1aa',
                fontWeight: 600,
              }}
            >
              Multi-Head Attention (MHA)
            </button>
            <button
              data-testid="tab-mlp"
              onClick={() => setActiveComponent('mlp')}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                background: activeComponent === 'mlp' ? '#3f3f46' : '#27272a',
                color: activeComponent === 'mlp' ? '#10b981' : '#a1a1aa',
                fontWeight: 600,
              }}
            >
              Feedforward (MLP)
            </button>
            <button
              data-testid="tab-ln"
              onClick={() => setActiveComponent('ln')}
              style={{
                padding: '6px 12px',
                borderRadius: '6px',
                border: 'none',
                cursor: 'pointer',
                background: activeComponent === 'ln' ? '#3f3f46' : '#27272a',
                color: activeComponent === 'ln' ? '#10b981' : '#a1a1aa',
                fontWeight: 600,
              }}
            >
              LayerNorm & Residual
            </button>
          </div>

          {activeComponent === 'attn' && (
            <div data-testid="mha-panel">
              <span style={{ fontSize: '13px', color: '#a1a1aa', display: 'block', marginBottom: '8px' }}>
                Layer {selectedLayer} Attention Heads ({numHeads} Heads):
              </span>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
                {Array.from({ length: numHeads }).map((_, h) => (
                  <div
                    key={h}
                    data-testid={`head-card-${h}`}
                    style={{ padding: '8px', background: '#27272a', borderRadius: '6px', textAlign: 'center', border: '1px solid #3f3f46' }}
                  >
                    <div style={{ fontWeight: 600, fontSize: '12px', color: '#10b981' }}>Head {h}</div>
                    <div style={{ fontSize: '11px', color: '#71717a' }}>Dim: {hiddenSize / numHeads}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeComponent === 'mlp' && (
            <div data-testid="mlp-panel" style={{ padding: '12px', background: '#27272a', borderRadius: '6px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, color: '#f4f4f5', marginBottom: '6px' }}>Layer {selectedLayer} MLP Sublayer</div>
              <div style={{ fontSize: '12px', color: '#a1a1aa', lineHeight: 1.6 }}>
                Projection: <code>{hiddenSize} → {hiddenSize * 4} → {hiddenSize}</code><br />
                Activation Function: <strong>GELU / SwiGLU</strong><br />
                Total Neurons: <strong>{hiddenSize * 4}</strong>
              </div>
            </div>
          )}

          {activeComponent === 'ln' && (
            <div data-testid="ln-panel" style={{ padding: '12px', background: '#27272a', borderRadius: '6px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, color: '#f4f4f5', marginBottom: '6px' }}>Layer {selectedLayer} Pre-LN & Residual Stream</div>
              <div style={{ fontSize: '12px', color: '#a1a1aa', lineHeight: 1.6 }}>
                Scale / Shift parameters: <code>{hiddenSize} dimensions</code><br />
                Residual Stream Addition: <code>x_out = x_in + Sublayer(LN(x_in))</code>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
