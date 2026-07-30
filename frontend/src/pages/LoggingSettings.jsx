import React from 'react';

export default function LoggingSettings({ settings, onChange }) {
  const log = settings?.logging || {};
  return (
    <div className="card" data-testid="settings-section-logging">
      <h3>Logging & Event System Settings</h3>
      <div className="field">
        <label>Log Level</label>
        <select
          value={log.level || 'info'}
          onChange={(e) => onChange({ logging: { ...log, level: e.target.value } })}
        >
          <option value="debug">Debug</option>
          <option value="info">Info</option>
          <option value="warn">Warn</option>
          <option value="error">Error</option>
        </select>
      </div>
    </div>
  );
}
