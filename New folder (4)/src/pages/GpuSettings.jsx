import React from 'react';

const ACCEL = [
  { value: 'auto', label: 'Auto' },
  { value: 'on', label: 'Force on' },
  { value: 'off', label: 'Disabled' }
];

export default function GpuSettings({ settings, onChange }) {
  const gpu = settings.gpu || { enabled: true, acceleration: 'auto' };
  return (
    <div className="card">
      <h2>GPU</h2>
      <p className="hint">Hardware acceleration preferences. Restart the app for changes to take effect.</p>
      <div className="row">
        <label>Enable GPU</label>
        <input
          type="checkbox"
          checked={gpu.enabled}
          onChange={(e) => onChange({ gpu: { ...gpu, enabled: e.target.checked } })}
        />
      </div>
      <div className="row">
        <label>Acceleration</label>
        <select
          value={gpu.acceleration}
          onChange={(e) => onChange({ gpu: { ...gpu, acceleration: e.target.value } })}
        >
          {ACCEL.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </div>
    </div>
  );
}
