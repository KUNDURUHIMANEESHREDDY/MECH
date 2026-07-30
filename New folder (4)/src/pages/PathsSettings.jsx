import React from 'react';

export default function PathsSettings({ settings, onChange }) {
  const paths = settings.paths || {};
  return (
    <div className="card">
      <h2>Paths</h2>
      <p className="hint">Override default executable and folder locations.</p>
      <div className="row">
        <label>Python executable</label>
        <input
          type="text"
          value={paths.python || ''}
          onChange={(e) => onChange({ paths: { ...paths, python: e.target.value } })}
        />
      </div>
      <div className="row">
        <label>Workspace folder</label>
        <input
          type="text"
          value={paths.workspace || ''}
          onChange={(e) => onChange({ paths: { ...paths, workspace: e.target.value } })}
        />
      </div>
      <div className="row">
        <label>Projects folder</label>
        <input
          type="text"
          value={paths.projects || ''}
          onChange={(e) => onChange({ paths: { ...paths, projects: e.target.value } })}
        />
      </div>
    </div>
  );
}
