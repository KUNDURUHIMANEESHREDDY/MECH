import React from 'react';

export default function CacheSettings({ settings, onChange }) {
  const cache = settings.cache || { enabled: true, maxSizeMb: 1024, location: '' };
  return (
    <div className="card">
      <h2>Cache</h2>
      <p className="hint">Configure local cache size and location.</p>
      <div className="row">
        <label>Enabled</label>
        <input
          type="checkbox"
          checked={cache.enabled}
          onChange={(e) => onChange({ cache: { ...cache, enabled: e.target.checked } })}
        />
      </div>
      <div className="row">
        <label>Max size (MB)</label>
        <input
          type="number"
          min="0"
          value={cache.maxSizeMb}
          onChange={(e) => onChange({ cache: { ...cache, maxSizeMb: Number(e.target.value) } })}
        />
      </div>
      <div className="row">
        <label>Location</label>
        <input
          type="text"
          value={cache.location}
          placeholder="(default)"
          onChange={(e) => onChange({ cache: { ...cache, location: e.target.value } })}
        />
      </div>
    </div>
  );
}
