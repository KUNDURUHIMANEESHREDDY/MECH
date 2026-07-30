import React, { useState } from 'react';

export default function CircuitExplorerView({ api }) {
  const [selectedCircuit, setSelectedCircuit] = useState('ioi_circuit');
  
  // Hardcoded UI structure based on user request:
  // Circuit -> Evidence -> Confidence -> Members -> Attention Heads -> Neurons -> SAE Features -> Patch -> Compare
  
  return (
    <div style={{ padding: 32, background: '#0b0b1a', color: '#e0e0ff', height: '100%', overflowY: 'auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 8px 0', color: '#d0c0ff' }}>Circuit Explorer</h1>
        <p style={{ margin: 0, color: '#888', fontSize: 14 }}>
          Hierarchical exploration of mechanistic circuits.
        </p>
      </div>

      <div style={{ display: 'flex', gap: 24 }}>
        {/* Left pane: Circuit selection & metadata */}
        <div style={{ flex: '1', background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 24 }}>
          <h2 style={{ fontSize: 16, color: '#fff', margin: '0 0 16px 0' }}>Circuit Selection</h2>
          
          <select 
            value={selectedCircuit}
            onChange={e => setSelectedCircuit(e.target.value)}
            style={{ 
              width: '100%', padding: '8px 12px', background: '#1a1a2e', 
              color: '#fff', border: '1px solid #3a3a5a', borderRadius: 6,
              marginBottom: 24
            }}
          >
            <option value="ioi_circuit">Indirect Object Identification (IOI)</option>
            <option value="induction_circuit">Induction Heads</option>
            <option value="greater_than_circuit">Greater-Than Circuit</option>
          </select>
          
          <div style={{ borderLeft: '2px solid #5cd4c4', paddingLeft: 16, marginBottom: 24 }}>
            <h3 style={{ fontSize: 13, color: '#888', margin: '0 0 8px 0', textTransform: 'uppercase' }}>Evidence</h3>
            <p style={{ fontSize: 14, margin: 0, color: '#ddd' }}>
              Path patching shows Name Mover Heads (L9H9, L10H0) directly write to the IO token logits.
            </p>
          </div>

          <div style={{ borderLeft: '2px solid #f5c518', paddingLeft: 16, marginBottom: 24 }}>
            <h3 style={{ fontSize: 13, color: '#888', margin: '0 0 8px 0', textTransform: 'uppercase' }}>Confidence</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{ flex: 1, background: '#2a2a4a', height: 8, borderRadius: 4 }}>
                <div style={{ width: '95%', background: '#f5c518', height: '100%', borderRadius: 4 }} />
              </div>
              <span style={{ fontSize: 14, color: '#f5c518', fontWeight: 700 }}>95%</span>
            </div>
          </div>
        </div>

        {/* Right pane: Drilldown */}
        <div style={{ flex: '2', background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 24 }}>
          <h2 style={{ fontSize: 16, color: '#fff', margin: '0 0 16px 0' }}>Members & Components</h2>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            
            {/* Attention Heads */}
            <div style={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8, padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <h3 style={{ fontSize: 14, margin: 0, color: '#c9ada7' }}>Attention Heads</h3>
                <span style={{ fontSize: 12, background: '#222', padding: '2px 8px', borderRadius: 4 }}>Name Movers</span>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <span style={{ background: '#333', padding: '4px 8px', borderRadius: 4, fontSize: 13 }}>L9H9</span>
                <span style={{ background: '#333', padding: '4px 8px', borderRadius: 4, fontSize: 13 }}>L10H0</span>
              </div>
            </div>

            {/* Neurons */}
            <div style={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8, padding: 16 }}>
              <h3 style={{ fontSize: 14, margin: '0 0 12px 0', color: '#f2e9e4' }}>Key MLP Neurons</h3>
              <div style={{ display: 'flex', gap: 8 }}>
                <span style={{ background: '#333', padding: '4px 8px', borderRadius: 4, fontSize: 13 }}>L8N1204</span>
              </div>
            </div>

            {/* SAE Features */}
            <div style={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8, padding: 16 }}>
              <h3 style={{ fontSize: 14, margin: '0 0 12px 0', color: '#2a9d8f' }}>SAE Features</h3>
              <div style={{ display: 'flex', gap: 8 }}>
                <span style={{ background: '#333', padding: '4px 8px', borderRadius: 4, fontSize: 13, borderLeft: '2px solid #2a9d8f' }}>Feature 451 (Syntax)</span>
              </div>
            </div>

            {/* Patch & Compare Actions */}
            <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
              <button style={{ 
                flex: 1, padding: 12, background: '#2a2a4a', color: '#fff', 
                border: '1px solid #4a4e69', borderRadius: 8, cursor: 'pointer',
                fontWeight: 600
              }}>
                Patch Activation
              </button>
              <button style={{ 
                flex: 1, padding: 12, background: '#2a2a4a', color: '#fff', 
                border: '1px solid #4a4e69', borderRadius: 8, cursor: 'pointer',
                fontWeight: 600
              }}>
                Cross-model Alignment
              </button>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}
