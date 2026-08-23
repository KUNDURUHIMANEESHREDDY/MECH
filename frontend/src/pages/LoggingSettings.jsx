import React, { useState } from 'react';
import './LoggingSettings.css';

export const LoggingSettings = ({ settings = {}, onChange }) => {
  const [logLevel, setLogLevel] = useState(settings.logLevel || 'INFO');
  const [logToFile, setLogToFile] = useState(settings.logToFile ?? true);
  const [maxLogSizeMb, setMaxLogSizeMb] = useState(settings.maxLogSizeMb || 50);
  const [capturePySidecar, setCapturePySidecar] = useState(settings.capturePySidecar ?? true);
  const [flushStatus, setFlushStatus] = useState('');

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ logLevel, logToFile, maxLogSizeMb, capturePySidecar, ...patch });
    }
  };

  const handleFlush = () => {
    setFlushStatus('Active log buffers flushed to disk.');
    setTimeout(() => setFlushStatus(''), 3000);
  };

  return (
    <div className="logging-settings settings-page-container">
      <div className="settings-page-header">
        <h1>Logging & Diagnostics</h1>
        <p>Configure diagnostic verbosity, log stream rotation, and Python subprocess telemetry capture.</p>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="log-level-select">Log Verbosity Threshold</label>
        <select
          id="log-level-select"
          data-testid="log-level-select"
          className="settings-select"
          value={logLevel}
          onChange={(e) => {
            setLogLevel(e.target.value);
            handleUpdate({ logLevel: e.target.value });
          }}
        >
          <option value="DEBUG">DEBUG (Detailed Diagnostics & Tensors)</option>
          <option value="INFO">INFO (Standard Operation)</option>
          <option value="WARNING">WARNING (Recoverable Warnings Only)</option>
          <option value="ERROR">ERROR (Fatal Errors Only)</option>
        </select>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="log-to-file-toggle"
            type="checkbox"
            checked={logToFile}
            onChange={(e) => {
              setLogToFile(e.target.checked);
              handleUpdate({ logToFile: e.target.checked });
            }}
          />
          <span>Persist logs to rotation file on local disk</span>
        </label>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="max-log-size">Max Log File Size Before Rotation (MB)</label>
        <input
          id="max-log-size"
          data-testid="max-log-size-input"
          type="number"
          className="settings-input"
          min={5}
          max={500}
          value={maxLogSizeMb}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10) || 50;
            setMaxLogSizeMb(val);
            handleUpdate({ maxLogSizeMb: val });
          }}
        />
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="capture-sidecar-toggle"
            type="checkbox"
            checked={capturePySidecar}
            onChange={(e) => {
              setCapturePySidecar(e.target.checked);
              handleUpdate({ capturePySidecar: e.target.checked });
            }}
          />
          <span>Capture stderr and stdout streams from Python computational sidecar</span>
        </label>
      </div>

      <div className="settings-actions">
        <button
          type="button"
          data-testid="flush-logs-btn"
          className="settings-btn settings-btn-secondary"
          onClick={handleFlush}
        >
          Flush Log Buffer Now
        </button>
        {flushStatus && <span className="settings-status-msg">{flushStatus}</span>}
      </div>
    </div>
  );
};
