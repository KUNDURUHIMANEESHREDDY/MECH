import React from 'react';
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
  { key: 'theme', label: 'Theme', component: ThemeSettings },
  { key: 'gpu', label: 'GPU', component: GpuSettings },
  { key: 'cache', label: 'Cache', component: CacheSettings },
  { key: 'paths', label: 'Paths', component: PathsSettings },
  { key: 'models', label: 'Models', component: ModelsSettings },
  { key: 'python', label: 'Python', component: PythonSettings },
  { key: 'performance', label: 'Performance', component: PerformanceSettings },
  { key: 'plugins', label: 'Plugins', component: PluginsSettings },
  { key: 'debugger', label: 'Debugger', component: DebuggerSettings },
  { key: 'logging', label: 'Logging', component: LoggingSettings },
];

export default function Settings({ settings, onChange, api }) {
  if (!settings) {
    return <div className="card"><p className="hint">Loading settings…</p></div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card" style={{ marginBottom: 0 }}>
        <h2>Settings</h2>
        <div className="toolbar" style={{ flexWrap: 'wrap', gap: '8px' }}>
          <button className="danger" onClick={() => onChange({ __reset: true })}>Reset to defaults</button>
        </div>
      </div>

      {SECTIONS.map(({ key, label, component: Component }) => (
        <Component key={key} settings={settings} onChange={onChange} />
      ))}
    </div>
  );
}

