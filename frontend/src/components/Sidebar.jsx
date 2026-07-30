import React from 'react';

const ICONS = {
  workspace: '🧭',
  models: '🧠',
  prompts: '💬',
  debugger: '⚡',
  experiments: '🧪',
  sessions: '🔬',
  reports: '📊',
  settings: '⚙️',
  logging: '📋',
  build: '🔨',
  neuralexplorer: '🌌',
  benchmark: '📈',
  knowledgegraph: '🕸️',
  circuitexplorer: '🔍',
  reasoning: '🔬',
  evidencefusion: '🏛️',
  campaigns: '🧪',
  analytics: '📊',
  health: '🏥',
  plugins: '🔌',
  benchmarksuite: '📐'
};

export default function Sidebar({ pages, active, onSelect }) {
  return (
    <nav className="sidebar" aria-label="Primary">
      <div className="sidebar-header">
        <h1>Neural Debugger</h1>
        <span className="version-badge">v0.2.0</span>
      </div>
      <div className="nav-group">
        {Object.entries(pages).map(([key, page]) => (
          <div
            key={key}
            role="button"
            tabIndex={0}
            className={'nav-item' + (active === key ? ' active' : '')}
            onClick={() => onSelect(key)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect(key); }}
            data-testid={`nav-${key}`}
          >
            <span className="icon" aria-hidden>{ICONS[key] || '•'}</span>
            <span>{page.label}</span>
          </div>
        ))}
      </div>
    </nav>
  );
}
