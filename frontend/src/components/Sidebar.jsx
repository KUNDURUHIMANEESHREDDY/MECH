import React, { useState } from 'react';
import {
  Network,
  Workflow,
  PanelsTopLeft,
  BrainCircuit,
  Share2,
  Waypoints,
  Boxes,
  Terminal,
  Bug,
  Hammer,
  Gauge,
  ListChecks,
  FlaskConical,
  History,
  FileText,
  GitBranch,
  Scale,
  TrendingUp,
  HeartPulse,
  Rocket,
  Puzzle,
  BookOpen,
  ClipboardList,
  CopyCheck,
  Settings,
  ScrollText,
  FolderKanban,
  Clock,
  Search,
  ChevronUp,
  Activity,
} from 'lucide-react';

const SECTIONS = [
  {
    title: 'Explore',
    items: ['explorer', 'gpt2', 'transformer', 'workspace', 'neuralexplorer', 'knowledgegraph', 'circuitexplorer'],
  },
  {
    title: 'Develop',
    items: ['models', 'prompts', 'debugger', 'build', 'benchmark', 'benchmarksuite'],
  },
  {
    title: 'Research',
    items: ['experiments', 'sessions', 'reports', 'reasoning', 'evidencefusion', 'analytics', 'health'],
  },
  {
    title: 'General',
    items: ['campaigns', 'plugins', 'notebook', 'labnotebook', 'reproduction', 'settings', 'logging', 'projects', 'recent'],
  },
];

const PAGE_LABELS = {
  explorer: 'Model Explorer',
  gpt2: 'GPT-2 Live',
  transformer: 'Transformer Visualizer',
  workspace: 'Workspace',
  neuralexplorer: 'Neural Explorer',
  knowledgegraph: 'Knowledge Graph',
  circuitexplorer: 'Circuit Explorer',
  models: 'Models',
  prompts: 'Prompts',
  debugger: 'Debugger',
  build: 'Build Log',
  benchmark: 'Benchmark',
  benchmarksuite: 'Benchmark Suite',
  experiments: 'Experiments',
  sessions: 'Sessions',
  reports: 'Reports',
  reasoning: 'Reasoning Trace',
  evidencefusion: 'Evidence Fusion',
  analytics: 'Analytics',
  health: 'Health',
  campaigns: 'Campaigns',
  plugins: 'Plugins',
  notebook: 'Research Notebook',
  labnotebook: 'Lab Notebook',
  reproduction: 'Paper Reproduction',
  settings: 'Settings',
  logging: 'Logging',
  projects: 'Projects',
  recent: 'Recent Files',
};

const ICONS = {
  explorer: Network,
  gpt2: Activity,
  transformer: Workflow,
  workspace: PanelsTopLeft,
  neuralexplorer: BrainCircuit,
  knowledgegraph: Share2,
  circuitexplorer: Waypoints,
  models: Boxes,
  prompts: Terminal,
  debugger: Bug,
  build: Hammer,
  benchmark: Gauge,
  benchmarksuite: ListChecks,
  experiments: FlaskConical,
  sessions: History,
  reports: FileText,
  reasoning: GitBranch,
  evidencefusion: Scale,
  analytics: TrendingUp,
  health: HeartPulse,
  campaigns: Rocket,
  plugins: Puzzle,
  notebook: BookOpen,
  labnotebook: ClipboardList,
  reproduction: CopyCheck,
  settings: Settings,
  logging: ScrollText,
  projects: FolderKanban,
  recent: Clock,
};

export default function Sidebar({ pages, active, onSelect }) {
  const [query, setQuery] = useState('');
  const getLabel = (key) => pages[key]?.label || PAGE_LABELS[key] || key;
  const matches = (key) => getLabel(key).toLowerCase().includes(query.trim().toLowerCase());

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1>Explorer</h1>
        <div className="sidebar-actions">
          <button className="btn btn-ghost btn-sm" title="Collapse all"><ChevronUp size={14} /></button>
        </div>
      </div>
      <div className="sidebar-body">
        <div className="sidebar-search">
          <Search size={13} />
          <input
            type="text"
            className="input-text"
            placeholder="Search views..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        {SECTIONS.map(section => (
          <div key={section.title}>
            <div className="nav-section-title">{section.title}</div>
            <div className="nav-group">
              {section.items.filter(matches).map(key => {
                const Icon = ICONS[key];
                return (
                  <div
                    key={key}
                    role="button"
                    tabIndex={0}
                    className={'nav-item' + (active === key ? ' active' : '')}
                    onClick={() => onSelect(key)}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onSelect(key); }}
                    data-testid={`nav-${key}`}
                  >
                    {Icon && <span className="nav-item-icon"><Icon size={15} strokeWidth={1.75} /></span>}
                    <span className="nav-item-label">{getLabel(key)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
