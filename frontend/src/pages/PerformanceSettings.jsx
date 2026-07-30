import React from 'react';

export default function PerformanceSettings({ settings, onChange }) {
  const perf = settings?.performance || {};
  return (
    <div className="card" data-testid="settings-section-performance">
      <h3>Performance Optimization</h3>
      <div className="field">
        <label>CPU Threads</label>
        <input
          type="number"
          value={perf.threads || 4}
          onChange={(e) => onChange({ performance: { ...perf, threads: Number(e.target.value) } })}
        />
      </div>
      <div className="field">
        <label>Memory Limit (GB)</label>
        <input
          type="number"
          value={perf.memoryLimitGb || 8}
          onChange={(e) => onChange({ performance: { ...perf, memoryLimitGb: Number(e.target.value) } })}
        />
      </div>
    </div>
  );
}
