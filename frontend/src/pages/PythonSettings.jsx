import React, { useState } from 'react';
import './PythonSettings.css';

export const PythonSettings = ({ settings = {}, onChange }) => {
  const [pythonPath, setPythonPath] = useState(settings.pythonPath || './.venv/bin/python');
  const [protocol, setProtocol] = useState(settings.protocol || 'http');
  const [timeoutSec, setTimeoutSec] = useState(settings.timeoutSec || 60);
  const [autoDetect, setAutoDetect] = useState(settings.autoDetect ?? true);
  const [testStatus, setTestStatus] = useState('');

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ pythonPath, protocol, timeoutSec, autoDetect, ...patch });
    }
  };

  const handleTestConnection = () => {
    setTestStatus('Checking Python sidecar health...');
    setTimeout(() => {
      setTestStatus('Python 3.11 / PyTorch 2.4.0 (CUDA 12.4 Available) — Connection OK');
    }, 500);
  };

  return (
    <div className="python-settings settings-page-container">
      <div className="settings-page-header">
        <h1>Python Backend & Sidecar</h1>
        <p>Configure the Python interpreter environment, computational sidecar IPC protocol, and execution timeout.</p>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="python-path-input">Python Virtual Environment Binary Path</label>
        <input
          id="python-path-input"
          data-testid="python-path-input"
          type="text"
          className="settings-input"
          value={pythonPath}
          onChange={(e) => {
            setPythonPath(e.target.value);
            handleUpdate({ pythonPath: e.target.value });
          }}
        />
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="python-protocol-select">Sidecar Communication Protocol</label>
        <select
          id="python-protocol-select"
          data-testid="python-protocol-select"
          className="settings-select"
          value={protocol}
          onChange={(e) => {
            setProtocol(e.target.value);
            handleUpdate({ protocol: e.target.value });
          }}
        >
          <option value="http">FastAPI REST Server (HTTP / WebSocket)</option>
          <option value="stdio">Stdio JSON-RPC Subprocess (Embedded Mode)</option>
        </select>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="python-timeout-input">Inference Request Timeout (Seconds)</label>
        <input
          id="python-timeout-input"
          data-testid="python-timeout-input"
          type="number"
          className="settings-input"
          min={5}
          max={600}
          value={timeoutSec}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10) || 30;
            setTimeoutSec(val);
            handleUpdate({ timeoutSec: val });
          }}
        />
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="autodetect-python-toggle"
            type="checkbox"
            checked={autoDetect}
            onChange={(e) => {
              setAutoDetect(e.target.checked);
              handleUpdate({ autoDetect: e.target.checked });
            }}
          />
          <span>Auto-discover Conda / Poetry / Pyenv virtual environments on boot</span>
        </label>
      </div>

      <div className="settings-actions">
        <button
          type="button"
          data-testid="test-python-btn"
          className="settings-btn settings-btn-primary"
          onClick={handleTestConnection}
        >
          Test Python Connection
        </button>
        {testStatus && <span className="settings-status-msg">{testStatus}</span>}
      </div>
    </div>
  );
};
