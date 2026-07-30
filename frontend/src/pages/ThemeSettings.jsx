import React from 'react';

const OPTIONS = [
  { value: 'system', label: 'Follow system' },
  { value: 'dark', label: 'Dark' },
  { value: 'light', label: 'Light' }
];

export default function ThemeSettings({ settings, onChange }) {
  return (
    <div className="card">
      <h2>Theme</h2>
      <p className="hint">Choose how the application is styled.</p>
      <div className="row">
        <label>Appearance</label>
        <select
          data-testid="theme-select"
          value={settings.theme}
          onChange={(e) => onChange({ theme: e.target.value })}
        >
          {OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </div>
    </div>
  );
}
