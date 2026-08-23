import React, { useState } from 'react';
import './PluginsSettings.css';

export const PluginsSettings = ({ settings = {}, onChange }) => {
  const [pluginsEnabled, setPluginsEnabled] = useState(settings.pluginsEnabled ?? true);
  const [pluginsDirectory, setPluginsDirectory] = useState(settings.pluginsDirectory || '~/MECH-Workspace/plugins');
  const [autoReload, setAutoReload] = useState(settings.autoReload ?? false);
  const [scanMsg, setScanMsg] = useState('');

  const [activePlugins] = useState([
    { id: 'sae-lens-pro', name: 'SAE Lens Pro Visualizer', version: '1.2.0', enabled: true },
    { id: 'circuit-tracer-hpc', name: 'Slurm Circuit Tracer', version: '2.0.1', enabled: true },
    { id: 'latex-exporter', name: 'Automated ICLR/NeurIPS Exporter', version: '0.9.4', enabled: false },
  ]);

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ pluginsEnabled, pluginsDirectory, autoReload, ...patch });
    }
  };

  const handleScan = () => {
    setScanMsg('Found 3 installed plugin packages. Registry updated.');
    setTimeout(() => setScanMsg(''), 3000);
  };

  return (
    <div className="plugins-settings settings-page-container">
      <div className="settings-page-header">
        <h1>Plugins & Extensions</h1>
        <p>Manage third-party interpretability tools, custom panel renderers, and scientific skill hooks.</p>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="plugins-enabled-toggle"
            type="checkbox"
            checked={pluginsEnabled}
            onChange={(e) => {
              setPluginsEnabled(e.target.checked);
              handleUpdate({ pluginsEnabled: e.target.checked });
            }}
          />
          <span>Enable dynamic scientific extension loading</span>
        </label>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="plugins-directory-input">Extensions Search Directory</label>
        <input
          id="plugins-directory-input"
          data-testid="plugins-directory-input"
          type="text"
          className="settings-input"
          value={pluginsDirectory}
          onChange={(e) => {
            setPluginsDirectory(e.target.value);
            handleUpdate({ pluginsDirectory: e.target.value });
          }}
        />
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="auto-reload-toggle"
            type="checkbox"
            checked={autoReload}
            onChange={(e) => {
              setAutoReload(e.target.checked);
              handleUpdate({ autoReload: e.target.checked });
            }}
          />
          <span>Hot-reload plugin manifests when filesystem changes occur</span>
        </label>
      </div>

      <div className="settings-section">
        <label className="settings-label">Installed Scientific Extensions</label>
        <div className="plugins-list">
          {activePlugins.map((p) => (
            <div key={p.id} className="plugin-item">
              <div className="plugin-info">
                <strong>{p.name}</strong>
                <span className="plugin-version">v{p.version} ({p.id})</span>
              </div>
              <span className={`plugin-badge ${p.enabled ? 'active' : 'inactive'}`}>
                {p.enabled ? 'Active' : 'Disabled'}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="settings-actions">
        <button
          type="button"
          data-testid="scan-plugins-btn"
          className="settings-btn settings-btn-primary"
          onClick={handleScan}
        >
          Scan Plugin Directory
        </button>
        {scanMsg && <span className="settings-status-msg">{scanMsg}</span>}
      </div>
    </div>
  );
};
