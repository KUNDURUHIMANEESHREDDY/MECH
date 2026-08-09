import React from 'react';
import { useAppStore } from '../store/useAppStore';
import { Link, useLocation } from 'react-router-dom';
import {
  Home,
  Cpu,
  Search,
  Settings,
  Play,
  Download,
  FolderTree,
  FlaskConical,
  BarChart3,
  FileText,
  Activity,
  Code,
  BookOpen,
  Brain,
  Layers,
  Zap,
  PanelLeft,
  Sidebar,
} from 'lucide-react';
import './Sidebar.css';

const PAGES = {
  workspace: { label: 'Workspace', icon: Home, path: '/workspace' },
  models: { label: 'Models', icon: Cpu, path: '/models' },
  explorer: { label: 'Model Explorer', icon: Search, path: '/explorer' },
  transformer: { label: 'Transformer Visualizer', icon: Layers, path: '/transformer' },
  neuralexplorer: { label: 'Neural Explorer', icon: Zap, path: '/neuralexplorer' },
  debugger: { label: 'Debugger', icon: Code, path: '/debugger' },
  experiments: { label: 'Experiments', icon: FlaskConical, path: '/experiments' },
  sessions: { label: 'Sessions', icon: Activity, path: '/sessions' },
  reports: { label: 'Reports', icon: FileText, path: '/reports' },
  settings: { label: 'Settings', icon: Settings, path: '/settings' },
  logging: { label: 'Execution Logs', icon: Activity, path: '/logging' },
  build: { label: 'Build Log', icon: Code, path: '/build' },
  benchmark: { label: 'Benchmark', icon: BarChart3, path: '/benchmark' },
  benchmarksuite: { label: 'Benchmark Suite', icon: BarChart3, path: '/benchmarksuite' },
  prompts: { label: 'Prompts', icon: FileText, path: '/prompts' },
  notebooks: { label: 'Research Notebook', icon: BookOpen, path: '/notebooks' },
  knowledgegraph: { label: 'Knowledge Graph', icon: Brain, path: '/knowledgegraph' },
  reasoning: { label: 'Reasoning Trace', icon: Activity, path: '/reasoning' },
  evidencefusion: { label: 'Evidence Fusion', icon: Layers, path: '/evidencefusion' },
  analytics: { label: 'Analytics', icon: BarChart3, path: '/analytics' },
  health: { label: 'Scientific Health', icon: Activity, path: '/health' },
  campaigns: { label: 'Campaigns', icon: FlaskConical, path: '/campaigns' },
  plugins: { label: 'Plugin Registry', icon: Layers, path: '/plugins' },
  reproduction: { label: 'Paper Reproduction', icon: BookOpen, path: '/reproduction' },
  projects: { label: 'Research Projects', icon: FolderTree, path: '/projects' },
  recent: { label: 'Recent Files', icon: FileText, path: '/recent' },
};

interface SidebarProps {
  pages?: Record<string, { label: string; component?: React.ComponentType; path?: string }>;
  active?: string;
  onSelect?: (key: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ pages: pagesProp, active, onSelect }) => {
  const store = useAppStore();
  const location = useLocation();
  const pages = pagesProp || PAGES;

  const activeKey = active || location.pathname.slice(1).split('/')[0] || 'workspace';

  return (
    <aside className={`sidebar ${store.sidebarCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        <Brain size={18} className="sidebar-logo" />
        {!store.sidebarCollapsed && <span className="sidebar-title">MECH</span>}
      </div>

      <nav className="sidebar-nav">
        {Object.entries(pages).map(([key, page]) => {
          const Icon = page.icon || Home;
          const isActive = activeKey === key;
          return (
            <button
              key={key}
              data-testid={`nav-${key}`}
              className={`nav-item ${isActive ? 'active' : ''}`}
              onClick={() => onSelect?.(key)}
              title={page.label}
            >
              <Icon size={16} />
              {!store.sidebarCollapsed && <span>{page.label}</span>}
            </button>
          );
        })}
      </nav>
    </aside>
  );
};
