import React, { useState } from 'react';
import './ModelsSettings.css';

export const ModelsSettings = ({ settings = {}, onChange }) => {
  const [modelPath, setModelPath] = useState(settings.modelPath || 'gpt2');
  const [hfToken, setHfToken] = useState(settings.hfToken || '');
  const [offlineMode, setOfflineMode] = useState(settings.offlineMode ?? false);
  const [autoDownloadWeights, setAutoDownloadWeights] = useState(settings.autoDownloadWeights ?? true);

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ modelPath, hfToken, offlineMode, autoDownloadWeights, ...patch });
    }
  };

  return (
    <div className="models-settings settings-page-container">
      <div className="settings-page-header">
        <h1>Model Architectures & Weights</h1>
        <p>Configure default language model targets, Hugging Face credentials, and local checkpoint repositories.</p>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="model-select">Active Default Model Target</label>
        <select
          id="model-select"
          data-testid="model-select"
          className="settings-select"
          value={modelPath}
          onChange={(e) => {
            setModelPath(e.target.value);
            handleUpdate({ modelPath: e.target.value });
          }}
        >
          <option value="gpt2">GPT-2 Small (124M Parameters, 12 Layers)</option>
          <option value="gpt2-medium">GPT-2 Medium (355M Parameters, 24 Layers)</option>
          <option value="gpt2-large">GPT-2 Large (774M Parameters, 36 Layers)</option>
          <option value="gpt2-xl">GPT-2 XL (1.5B Parameters, 48 Layers)</option>
          <option value="google/gemma-2-2b">Gemma 2 2B (Google)</option>
          <option value="meta-llama/Llama-3.2-1B">LLaMA 3.2 1B (Meta)</option>
          <option value="Qwen/Qwen2.5-1.5B">Qwen 2.5 1.5B (Alibaba)</option>
          <option value="mistralai/Mistral-7B-v0.1">Mistral 7B (Mistral AI)</option>
          <option value="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">DeepSeek R1 Distill 1.5B</option>
        </select>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="hf-token-input">Hugging Face User Access Token</label>
        <input
          id="hf-token-input"
          data-testid="hf-token-input"
          type="password"
          className="settings-input"
          placeholder="hf_..."
          value={hfToken}
          onChange={(e) => {
            setHfToken(e.target.value);
            handleUpdate({ hfToken: e.target.value });
          }}
        />
        <small className="settings-hint">Required for gated models such as LLaMA-3 and Gemma-2.</small>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="offline-mode-toggle"
            type="checkbox"
            checked={offlineMode}
            onChange={(e) => {
              setOfflineMode(e.target.checked);
              handleUpdate({ offlineMode: e.target.checked });
            }}
          />
          <span>Offline / Air-Gapped Mode (Strictly use locally cached weights without querying HF Hub)</span>
        </label>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="auto-download-toggle"
            type="checkbox"
            checked={autoDownloadWeights}
            onChange={(e) => {
              setAutoDownloadWeights(e.target.checked);
              handleUpdate({ autoDownloadWeights: e.target.checked });
            }}
          />
          <span>Automatically download missing model shards on first prompt execution</span>
        </label>
      </div>
    </div>
  );
};
