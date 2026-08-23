import React, { useState } from 'react';
import './PathsSettings.css';

export const PathsSettings = ({ settings = {}, onChange }) => {
  const [workspacePath, setWorkspacePath] = useState(settings.workspacePath || '~/MECH-Workspace');
  const [exportsPath, setExportsPath] = useState(settings.exportsPath || '~/MECH-Workspace/exports');
  const [checkpointsPath, setCheckpointsPath] = useState(settings.checkpointsPath || '~/MECH-Workspace/checkpoints');
  const [statusMsg, setStatusMsg] = useState('');

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ workspacePath, exportsPath, checkpointsPath, ...patch });
    }
  };

  const handleReset = () => {
    const def = {
      workspacePath: '~/MECH-Workspace',
      exportsPath: '~/MECH-Workspace/exports',
      checkpointsPath: '~/MECH-Workspace/checkpoints',
    };
    setWorkspacePath(def.workspacePath);
    setExportsPath(def.exportsPath);
    setCheckpointsPath(def.checkpointsPath);
    handleUpdate(def);
    setStatusMsg('Paths reset to default.');
    setTimeout(() => setStatusMsg(''), 3000);
  };

  return (
    <div className="paths-settings settings-page-container">
      <div className="settings-page-header">
        <h1>Workspace & File Paths</h1>
        <p>Configure local directory paths for scientific workspaces, exported reports, and experiment snapshots.</p>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="workspace-path-input">Primary Research Workspace Path</label>
        <input
          id="workspace-path-input"
          data-testid="workspace-path-input"
          type="text"
          className="settings-input"
          value={workspacePath}
          onChange={(e) => {
            setWorkspacePath(e.target.value);
            handleUpdate({ workspacePath: e.target.value });
          }}
        />
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="exports-path-input">Publication & Export Artifacts Path</label>
        <input
          id="exports-path-input"
          data-testid="exports-path-input"
          type="text"
          className="settings-input"
          value={exportsPath}
          onChange={(e) => {
            setExportsPath(e.target.value);
            handleUpdate({ exportsPath: e.target.value });
          }}
        />
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="checkpoints-path-input">Experiment Checkpoints Directory</label>
        <input
          id="checkpoints-path-input"
          data-testid="checkpoints-path-input"
          type="text"
          className="settings-input"
          value={checkpointsPath}
          onChange={(e) => {
            setCheckpointsPath(e.target.value);
            handleUpdate({ checkpointsPath: e.target.value });
          }}
        />
      </div>

      <div className="settings-actions">
        <button
          type="button"
          data-testid="reset-paths-btn"
          className="settings-btn settings-btn-secondary"
          onClick={handleReset}
        >
          Reset Paths to Default
        </button>
        {statusMsg && <span className="settings-status-msg">{statusMsg}</span>}
      </div>
    </div>
  );
};
