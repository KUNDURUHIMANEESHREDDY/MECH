import React from 'react';

export default function PythonSettings({ settings, onChange }) {
  const py = settings?.python || {};
  return (
    <div className="card" data-testid="settings-section-python">
      <h3>Python Runtime Settings</h3>
      <div className="field">
        <label>Python Binary Path</label>
        <input
          type="text"
          value={py.path || ''}
          onChange={(e) => onChange({ python: { ...py, path: e.target.value } })}
          placeholder="e.g. python or C:\Python311\python.exe"
        />
      </div>
      <div className="field">
        <label>Environment Mode</label>
        <select
          value={py.environment || 'system'}
          onChange={(e) => onChange({ python: { ...py, environment: e.target.value } })}
        >
          <option value="system">System Default</option>
          <option value="venv">Virtual Environment (venv)</option>
          <option value="conda">Conda Environment</option>
        </select>
      </div>
    </div>
  );
}
