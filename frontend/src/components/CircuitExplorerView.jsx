import React, { useState } from 'react';
import { colors } from '../design/tokens/colors';

export default function CircuitExplorerView({ api }) {
  const [selectedCircuit, setSelectedCircuit] = useState('ioi_circuit');

  return (
    <div className="explorer-page">
      <div className="explorer-header">
        <h1>Circuit Explorer</h1>
        <p className="hint">Hierarchical exploration of mechanistic circuits.</p>
      </div>

      <div className="explorer-split">
        <div className="explorer-panel">
          <h2>Circuit Selection</h2>

          <select
            value={selectedCircuit}
            onChange={e => setSelectedCircuit(e.target.value)}
            className="input-text"
            style={{ marginBottom: 24 }}
          >
            <option value="ioi_circuit">Indirect Object Identification (IOI)</option>
            <option value="induction_circuit">Induction Heads</option>
            <option value="greater_than_circuit">Greater-Than Circuit</option>
          </select>

          <div className="evidence-block" style={{ borderLeftColor: colors.primary }}>
            <div className="evidence-label">Evidence</div>
            <p className="evidence-text">
              Path patching shows Name Mover Heads (L9H9, L10H0) directly write to the IO token logits.
            </p>
          </div>

          <div className="evidence-block" style={{ borderLeftColor: colors.warning }}>
            <div className="evidence-label">Confidence</div>
            <div className="confidence-bar">
              <div className="confidence-track">
                <div className="confidence-fill" style={{ width: '95%', background: colors.warning }} />
              </div>
              <span className="confidence-value">95%</span>
            </div>
          </div>
        </div>

        <div className="explorer-panel" style={{ flex: 2 }}>
          <h2>Members & Components</h2>

          <div className="component-list">
            <div className="component-card">
              <div className="component-header">
                <h3 style={{ color: colors.bodyMuted }}>Attention Heads</h3>
                <span className="component-tag">Name Movers</span>
              </div>
              <div className="component-tags">
                <span className="chip">L9H9</span>
                <span className="chip">L10H0</span>
              </div>
            </div>

            <div className="component-card">
              <h3 style={{ color: colors.ink, margin: '0 0 12px 0' }}>Key MLP Neurons</h3>
              <div className="component-tags">
                <span className="chip">L8N1204</span>
              </div>
            </div>

            <div className="component-card">
              <h3 style={{ color: colors.primary, margin: '0 0 12px 0' }}>SAE Features</h3>
              <div className="component-tags">
                <span className="chip" style={{ borderLeft: `2px solid ${colors.primary}` }}>Feature 451 (Syntax)</span>
              </div>
            </div>

            <div className="action-row">
              <button className="btn explorer-action-btn">Patch Activation</button>
              <button className="btn explorer-action-btn">Cross-model Alignment</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
