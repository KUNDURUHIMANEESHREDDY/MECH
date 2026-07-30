import React from 'react';

export default function PluginsSettings({ settings, onChange }) {
  const plug = settings?.plugins || {};
  return (
    <div className="card" data-testid="settings-section-plugins">
      <h3>Plugins & Extensions</h3>
      <div className="field">
        <label>
          <input
            type="checkbox"
            checked={Boolean(plug.autoUpdate)}
            onChange={(e) => onChange({ plugins: { ...plug, autoUpdate: e.target.checked } })}
          /> Automatically update research plugins
        </label>
      </div>
    </div>
  );
}
