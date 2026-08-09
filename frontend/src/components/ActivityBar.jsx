import React from 'react';
import { useAppStore } from '../store/useAppStore';
import {
  Cpu,
  Search,
  Settings,
  FlaskConical,
  BarChart3,
  FileText,
  Activity,
  Code,
  BookOpen,
  Brain,
  Layers,
  Zap,
} from 'lucide-react';
import './ActivityBar.css';

const ICONS: Record<string, React.ReactNode> = {
  workspace: <Layers size={18} />,
  models: <Cpu size={18} />,
  explorer: <Search size={18} />,
  transformer: <Layers size={18} />,
  neuralexplorer: <Zap size={18} />,
  debugger: <Code size={18} />,
  experiments: <FlaskConical size={18} />,
  sessions: <Activity size={18} />,
  reports: <FileText size={18} />,
  settings: <Settings size={18} />,
  logging: <Activity size={18} />,
  build: <Code size={18} />,
  benchmark: <BarChart3 size={18} />,
  benchmarksuite: <BarChart3 size={18} />,
  prompts: <FileText size={18} />,
  notebooks: <BookOpen size={18} />,
  knowledgegraph: <Brain size={18} />,
  reasoning: <Activity size={18} />,
  evidencefusion: <Layers size={18} />,
  analytics: <BarChart3 size={18} />,
  health: <Activity size={18} />,
  campaigns: <FlaskConical size={18} />,
  plugins: <Layers size={18} />,
  reproduction: <BookOpen size={18} />,
  projects: <Layers size={18} />,
  recent: <FileText size={18} />,
};

export const ActivityBar: React.FC = () => {
  const { activePage, setActivePage, activityCollapsed, toggleActivity } = useAppStore();

  if (activityCollapsed) {
    return (
      <aside className="activity-bar collapsed">
        <div className="activity-bar-icons">
          {Object.keys(ICONS).slice(0, 6).map((key) => (
            <button
              key={key}
              className={`activity-icon-btn ${activePage === key ? 'active' : ''}`}
              onClick={() => setActivePage(key)}
              title={key}
            >
              {ICONS[key]}
            </button>
          ))}
        </div>
      </aside>
    );
  }

  return (
    <aside className="activity-bar">
      <div className="activity-bar-header">
        <Brain size={18} className="activity-logo" />
        {!activityCollapsed && <span className="activity-title">MECH</span>}
      </div>

      <div className="activity-bar-icons">
        {Object.entries(ICONS).map(([key, icon]) => (
          <button
            key={key}
            className={`activity-icon-btn ${activePage === key ? 'active' : ''}`}
            onClick={() => setActivePage(key)}
            title={key}
          >
            {icon}
          </button>
        ))}
      </div>

      <div className="activity-bar-footer">
        <button className="activity-icon-btn" onClick={toggleActivity} title="Collapse">
          <Settings size={16} />
        </button>
      </div>
    </aside>
  );
};
