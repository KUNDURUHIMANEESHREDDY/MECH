import React from 'react';

export default function ModelsSettings({ settings, onChange }) {
  const models = settings?.models || {};
  return (
    <div className="card" data-testid="settings-section-models">
      <h3>Models Configuration</h3>
      <div className="field">
        <label>Default Model</label>
        <select
          value={models.defaultModel || 'GPT-2 Small'}
          onChange={(e) => onChange({ models: { ...models, defaultModel: e.target.value } })}
        >
          <option value="GPT-2 Small">GPT-2 Small</option>
          <option value="GPT-2 Medium">GPT-2 Medium</option>
          <option value="Pythia 160M">Pythia 160M</option>
          <option value="Llama-3 8B">Llama-3 8B</option>
        </select>
      </div>
      <div className="field">
        <label>Compute Device</label>
        <select
          value={models.device || 'cpu'}
          onChange={(e) => onChange({ models: { ...models, device: e.target.value } })}
        >
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA GPU</option>
          <option value="mps">Apple MPS</option>
        </select>
      </div>
    </div>
  );
}
