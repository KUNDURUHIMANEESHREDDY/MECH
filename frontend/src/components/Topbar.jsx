import React, { useState, useEffect } from 'react';
import {
  Search,
  Cpu,
  Play,
  Download,
  FolderTree,
  Settings,
  User,
  Activity,
  Layers,
  Sparkles,
  PanelLeft,
  Sidebar,
} from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { Link, useLocation } from 'react-router-dom';
import { CommandPalette } from './CommandPalette';
import './Topbar.css';

const PAGES = {
  workspace: { label: 'Workspace', path: '/workspace' },
  models: { label: 'Models', path: '/models' },
  explorer: { label: 'Model Explorer', path: '/explorer' },
  transformer: { label: 'Transformer Visualizer', path: '/transformer' },
  neuralexplorer: { label: 'Neural Explorer', path: '/neuralexplorer' },
  debugger: { label: 'Debugger', path: '/debugger' },
  experiments: { label: 'Experiments', path: '/experiments' },
  sessions: { label: 'Sessions', path: '/sessions' },
  reports: { label: 'Reports', path: '/reports' },
  settings: { label: 'Settings', path: '/settings' },
  logging: { label: 'Execution Logs', path: '/logging' },
  build: { label: 'Build Log', path: '/build' },
  benchmark: { label: 'Benchmark', path: '/benchmark' },
  benchmarksuite: { label: 'Benchmark Suite', path: '/benchmarksuite' },
  prompts: { label: 'Prompts', path: '/prompts' },
  notebooks: { label: 'Research Notebook', path: '/notebooks' },
  knowledgegraph: { label: 'Knowledge Graph', path: '/knowledgegraph' },
  reasoning: { label: 'Reasoning Trace', path: '/reasoning' },
  evidencefusion: { label: 'Evidence Fusion', path: '/evidencefusion' },
  analytics: { label: 'Analytics', path: '/analytics' },
  health: { label: 'Scientific Health', path: '/health' },
  campaigns: { label: 'Campaigns', path: '/campaigns' },
  plugins: { label: 'Plugin Registry', path: '/plugins' },
  reproduction: { label: 'Paper Reproduction', path: '/reproduction' },
  projects: { label: 'Research Projects', path: '/projects' },
  recent: { label: 'Recent Files', path: '/recent' },
};

export const Topbar: React.FC = () => {
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const location = useLocation();
  const { activeModel, activePage, setActivePage, sidebarCollapsed, toggleSidebar } = useAppStore();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const currentPage = PAGES[activePage] || PAGES.workspace;
  const breadcrumbs = [currentPage.label];

  return (
    <header className="topbar">
      <div className="topbar-left">
        <button onClick={toggleSidebar} className="topbar-icon-btn" title="Toggle Sidebar">
          <Sidebar size={18} />
        </button>
        <div className="topbar-breadcrumbs">
          {breadcrumbs.map((crumb, idx) => (
            <span key={idx} className={`topbar-crumb ${idx === breadcrumbs.length - 1 ? 'active' : ''}`}>
              {crumb}
            </span>
          ))}
        </div>
      </div>

      <div className="topbar-center">
        <button className="topbar-search-btn" onClick={() => setCommandPaletteOpen(true)}>
          <Search size={14} />
          <span>Search resources, panels, commands...</span>
          <kbd className="topbar-kbd">⌘K</kbd>
        </button>
      </div>

      <div className="topbar-right">
        <div className="topbar-model-select">
          <Cpu size={14} />
          <span>{activeModel || 'No Model'}</span>
        </div>

        <button className="topbar-action-btn" title="Run Activation Pass">
          <Play size={14} />
          <span>Run Pass</span>
        </button>

        <button className="topbar-icon-btn" title="Export Analysis State">
          <Download size={14} />
        </button>

        <div className="topbar-python-status" data-testid="python-status">
          <Activity size={12} />
          <span>Python Ready</span>
        </div>

        <button className="topbar-icon-btn" title="Workspace Settings">
          <Settings size={15} />
        </button>

        <button className="topbar-icon-btn" title="User Profile">
          <User size={15} />
        </button>
      </div>

      <CommandPalette open={commandPaletteOpen} onOpenChange={setCommandPaletteOpen} />
    </header>
  );
};
