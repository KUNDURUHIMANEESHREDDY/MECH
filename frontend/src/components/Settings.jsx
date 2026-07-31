import React, { useState } from 'react';
import ThemeSettings from '../pages/ThemeSettings.jsx';
import GpuSettings from '../pages/GpuSettings.jsx';
import CacheSettings from '../pages/CacheSettings.jsx';
import PathsSettings from '../pages/PathsSettings.jsx';
import ModelsSettings from '../pages/ModelsSettings.jsx';
import PythonSettings from '../pages/PythonSettings.jsx';
import PerformanceSettings from '../pages/PerformanceSettings.jsx';
import PluginsSettings from '../pages/PluginsSettings.jsx';
import DebuggerSettings from '../pages/DebuggerSettings.jsx';
import LoggingSettings from '../pages/LoggingSettings.jsx';

const SECTIONS = [
  { key: 'theme', label: 'Theme' },
  { key: 'gpu', label: 'GPU' },
  { key: 'cache', label: 'Cache' },
  { key: 'paths', label: 'Paths' },
  { key: 'models', label: 'Models' },
  { key: 'python', label: 'Python' },
  { key: 'performance', label: 'Performance' },
  { key: 'plugins', label: 'Plugins' },
  { key: 'debugger', label: 'Debugger' },
  { key: 'logging', label: 'Logging' }
];

export default function Settings({ settings, onChange, api }) {
  const [section, setSection] = useState('theme');
  if (!settings) {
    return <div className="card"><p className="hint">Loading settings…</p></div>;
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: '16px' }}>
        <h2>Settings</h2>
        <div className="toolbar" style={{ flexWrap: 'wrap', gap: '8px' }}>
          {SECTIONS.map((s) => (
            <button
              key={s.key}
              className={section === s.key ? '' : 'secondary'}
              onClick={() => setSection(s.key)}
              data-testid={`settings-tab-${s.key}`}
            >
              {s.label}
            </button>
          ))}
          <span style={{ flex: 1 }} />
          <button className="danger" onClick={() => onChange({ __reset: true })}>Reset to defaults</button>
        </div>
      </div>

      {section === 'theme' && <ThemeSettings settings={settings} onChange={onChange} />}
      {section === 'gpu' && <GpuSettings settings={settings} onChange={onChange} />}
      {section === 'cache' && <CacheSettings settings={settings} onChange={onChange} />}
      {section === 'paths' && <PathsSettings settings={settings} onChange={onChange} />}
      {section === 'models' && <ModelsSettings settings={settings} onChange={onChange} />}
      {section === 'python' && <PythonSettings settings={settings} onChange={onChange} />}
      {section === 'performance' && <PerformanceSettings settings={settings} onChange={onChange} />}
      {section === 'plugins' && <PluginsSettings settings={settings} onChange={onChange} />}
      {section === 'debugger' && <DebuggerSettings settings={settings} onChange={onChange} />}
      {section === 'logging' && <LoggingSettings settings={settings} onChange={onChange} />}
    </div>
  );
}

