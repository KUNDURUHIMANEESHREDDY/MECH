import React from 'react';

export default function DebuggerSettings({ settings, onChange }) {
  const dbg = settings?.debugger || {};
  return (
    <div className="card" data-testid="settings-section-debugger">
      <h3>Debugger Engine Settings</h3>
      <div className="field">
        <label>Token Highlighting Mode</label>
        <select
          value={dbg.tokenHighlighting || 'activation'}
          onChange={(e) => onChange({ debugger: { ...dbg, tokenHighlighting: e.target.value } })}
        >
          <option value="activation">Activation Magnitude</option>
          <option value="gradient">Gradient Attribution</option>
          <option value="entropy">Entropy</option>
        </select>
      </div>
    </div>
  );
}
