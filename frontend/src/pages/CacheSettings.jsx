import React, { useState } from 'react';
import './CacheSettings.css';

export const CacheSettings = ({ settings = {}, onChange }) => {
  const [cachePath, setCachePath] = useState(settings.cachePath || '~/.cache/neural-debugger');
  const [maxCacheSizeMb, setMaxCacheSizeMb] = useState(settings.maxCacheSizeMb || 2048);
  const [autoPurge, setAutoPurge] = useState(settings.autoPurge ?? true);
  const [purgePolicy, setPurgePolicy] = useState(settings.purgePolicy || 'LRU');
  const [statusMsg, setStatusMsg] = useState('');

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ cachePath, maxCacheSizeMb, autoPurge, purgePolicy, ...patch });
    }
  };

  const handlePurge = () => {
    setStatusMsg('Cache purged successfully.');
    setTimeout(() => setStatusMsg(''), 3000);
  };

  return (
    <div className="cache-settings settings-page-container">
      <div className="settings-page-header">
        <h1>Cache Settings</h1>
        <p>Configure on-disk tensor caching, activation shard retention, and automated purging policies.</p>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="cache-path-input">Cache Directory Path</label>
        <div className="settings-row">
          <input
            id="cache-path-input"
            data-testid="cache-path-input"
            type="text"
            className="settings-input"
            value={cachePath}
            onChange={(e) => {
              setCachePath(e.target.value);
              handleUpdate({ cachePath: e.target.value });
            }}
            placeholder="~/.cache/neural-debugger"
          />
        </div>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="max-cache-size">Maximum Disk Cache Size (MB)</label>
        <div className="settings-row">
          <input
            id="max-cache-size"
            data-testid="max-cache-size"
            type="number"
            className="settings-input"
            min={256}
            max={65536}
            value={maxCacheSizeMb}
            onChange={(e) => {
              const val = parseInt(e.target.value, 10) || 1024;
              setMaxCacheSizeMb(val);
              handleUpdate({ maxCacheSizeMb: val });
            }}
          />
        </div>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="cache-purge-policy">Eviction Policy</label>
        <select
          id="cache-purge-policy"
          data-testid="cache-purge-policy"
          className="settings-select"
          value={purgePolicy}
          onChange={(e) => {
            setPurgePolicy(e.target.value);
            handleUpdate({ purgePolicy: e.target.value });
          }}
        >
          <option value="LRU">Least Recently Used (LRU)</option>
          <option value="FIFO">First In, First Out (FIFO)</option>
          <option value="SIZE">Largest Shard First</option>
        </select>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="auto-purge-toggle"
            type="checkbox"
            checked={autoPurge}
            onChange={(e) => {
              setAutoPurge(e.target.checked);
              handleUpdate({ autoPurge: e.target.checked });
            }}
          />
          <span>Enable automated cache cleanup when disk space threshold is exceeded</span>
        </label>
      </div>

      <div className="settings-actions">
        <button
          type="button"
          data-testid="clear-cache-btn"
          className="settings-btn settings-btn-danger"
          onClick={handlePurge}
        >
          Clear Disk Cache Now
        </button>
        {statusMsg && <span className="settings-status-msg">{statusMsg}</span>}
      </div>
    </div>
  );
};
