import React, { useState } from 'react';
import { ChevronRight, ChevronDown, Folder, FileText, Brain, Layers, Flame, Zap, Cpu, Bookmark, Activity, Code, Database, BarChart, BookOpen, FlaskConical, Search, FolderTree } from 'lucide-react';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import { useUIStore } from '../../shared/stores/ui';
import { Resource, ResourceKind } from '../../shared/types';
import { pluginRegistry } from '../../panel-system/pluginRegistry';

interface NavTreeItem {
  id: string;
  label: string;
  kind: ResourceKind;
  icon: React.ReactNode;
  navKey?: string;
  children?: NavTreeItem[];
}

const RESOURCE_TREE: NavTreeItem[] = [
  {
    id: 'group_models',
    label: 'Models & Architectures',
    kind: 'model',
    icon: <Brain size={15} />,
    children: [
      { id: 'gpt2', label: 'GPT-2 Live', kind: 'model', icon: <Cpu size={14} />, navKey: 'gpt2' },
      { id: 'explorer', label: 'Model Explorer', kind: 'model', icon: <Search size={14} />, navKey: 'explorer' },
      { id: 'transformer', label: 'Transformer Visualizer', kind: 'model', icon: <Layers size={14} />, navKey: 'transformer' },
      { id: 'neuralexplorer', label: 'Neural Explorer', kind: 'neuron', icon: <Zap size={14} />, navKey: 'neuralexplorer' },
    ],
  },
  {
    id: 'group_analysis',
    label: 'Mechanistic Analysis',
    kind: 'circuit',
    icon: <Flame size={15} />,
    children: [
      { id: 'circuitexplorer', label: 'Circuit Explorer', kind: 'circuit', icon: <Activity size={14} />, navKey: 'circuitexplorer' },
      { id: 'knowledgegraph', label: 'Knowledge Graph', kind: 'circuit', icon: <Folder size={14} />, navKey: 'knowledgegraph' },
      { id: 'debugger', label: 'Debugger', kind: 'neuron', icon: <Code size={14} />, navKey: 'debugger' },
      { id: 'benchmark', label: 'Benchmark', kind: 'experiment', icon: <BarChart size={14} />, navKey: 'benchmark' },
      { id: 'benchmarksuite', label: 'Benchmark Suite', kind: 'experiment', icon: <BarChart size={14} />, navKey: 'benchmarksuite' },
      { id: 'experiments', label: 'Experiments', kind: 'experiment', icon: <FlaskConical size={14} />, navKey: 'experiments' },
      { id: 'reasoning', label: 'Reasoning', kind: 'session', icon: <FileText size={14} />, navKey: 'reasoning' },
      { id: 'evidencefusion', label: 'Evidence Fusion', kind: 'session', icon: <FileText size={14} />, navKey: 'evidencefusion' },
      { id: 'analytics', label: 'Analytics', kind: 'session', icon: <BarChart size={14} />, navKey: 'analytics' },
      { id: 'health', label: 'Health', kind: 'session', icon: <Activity size={14} />, navKey: 'health' },
      { id: 'campaigns', label: 'Campaigns', kind: 'session', icon: <FlaskConical size={14} />, navKey: 'campaigns' },
    ],
  },
  {
    id: 'group_workspace',
    label: 'Workspace & Assets',
    kind: 'workspace',
    icon: <Folder size={15} />,
    children: [
      { id: 'workspace', label: 'Workspace', kind: 'workspace', icon: <Folder size={14} />, navKey: 'workspace' },
      { id: 'models', label: 'Models', kind: 'model', icon: <Brain size={14} />, navKey: 'models' },
      { id: 'prompts', label: 'Prompts', kind: 'prompt', icon: <FileText size={14} />, navKey: 'prompts' },
      { id: 'build', label: 'Build', kind: 'workspace', icon: <Code size={14} />, navKey: 'build' },
      { id: 'sessions', label: 'Sessions', kind: 'session', icon: <FileText size={14} />, navKey: 'sessions' },
      { id: 'reports', label: 'Reports', kind: 'paper', icon: <BookOpen size={14} />, navKey: 'reports' },
      { id: 'plugins', label: 'Plugins', kind: 'workspace', icon: <Layers size={14} />, navKey: 'plugins' },
      { id: 'notebook', label: 'Research Notebook', kind: 'note', icon: <FileText size={14} />, navKey: 'notebook' },
      { id: 'labnotebook', label: 'Lab Notebook', kind: 'note', icon: <FileText size={14} />, navKey: 'labnotebook' },
      { id: 'reproduction', label: 'Paper Reproduction', kind: 'paper', icon: <BookOpen size={14} />, navKey: 'reproduction' },
      { id: 'settings', label: 'Settings', kind: 'workspace', icon: <Folder size={14} />, navKey: 'settings' },
      { id: 'logging', label: 'Logging', kind: 'session', icon: <Activity size={14} />, navKey: 'logging' },
      { id: 'projects', label: 'Projects', kind: 'workspace', icon: <Folder size={14} />, navKey: 'projects' },
      { id: 'recent', label: 'Recent Files', kind: 'session', icon: <FileText size={14} />, navKey: 'recent' },
    ],
  },
];

export const Navigator: React.FC = () => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    group_models: true,
    group_analysis: true,
    group_workspace: true,
  });

  const { openResource, openPanel } = useWorkspaceStore();
  const setCrumb = useUIStore((s) => s.setCrumb);

  const toggleExpand = (id: string) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleDoubleClick = (item: NavTreeItem) => {
    const res: Resource = {
      id: item.id,
      kind: item.kind,
      label: item.label,
    };
    const plugins = pluginRegistry.getByResourceKind(item.kind);
    const targetPanel = plugins.length > 0 ? plugins[0].id : 'token_viewer';
    openResource(res, targetPanel);
  };

  const handleNavClick = (item: NavTreeItem) => {
    if (item.navKey) {
      window.location.hash = `#${item.navKey}`;
    }
    if (item.label) {
      setCrumb(item.label);
    }
  };

  return (
    <aside className="shell-navigator" style={navStyle}>
      <div style={headerStyle}>
        <FolderTree size={14} style={{ color: 'var(--color-primary, #0066cc)' }} />
        <span style={{ fontWeight: 600, fontSize: '12px', letterSpacing: '-0.2px' }}>Resource Navigator</span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '6px 4px' }}>
        {RESOURCE_TREE.map((group) => {
          const isExp = expanded[group.id];
          return (
            <div key={group.id} style={{ marginBottom: '4px' }}>
              <div
                onClick={() => toggleExpand(group.id)}
                style={groupItemStyle}
              >
                {isExp ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                {group.icon}
                <span style={{ fontWeight: 600 }}>{group.label}</span>
              </div>

              {isExp && group.children && (
                <div style={{ paddingLeft: '16px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  {group.children.map((child) => (
                    <div
                      key={child.id}
                      data-testid={child.navKey ? `nav-${child.navKey}` : undefined}
                      onClick={() => handleNavClick(child)}
                      onDoubleClick={() => handleDoubleClick(child)}
                      style={childItemStyle}
                      title={`Double-click to open ${child.label} in Canvas`}
                    >
                      {child.icon}
                      <span style={{ flex: 1, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                        {child.label}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
};

const navStyle: React.CSSProperties = {
  width: 'var(--sidebar-width, 240px)',
  background: 'var(--bg-sidebar, #fafafa)',
  borderRight: '1px solid var(--border, #e0e0e0)',
  display: 'flex',
  flexDirection: 'column',
  userSelect: 'none',
  fontSize: '12px',
};

const headerStyle: React.CSSProperties = {
  height: '36px',
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '0 12px',
  borderBottom: '1px solid var(--border-light, #f0f0f0)',
  color: 'var(--text, #1d1d1f)',
};

const groupItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  padding: '6px 8px',
  borderRadius: '6px',
  color: 'var(--text-dim, #333333)',
  cursor: 'pointer',
  fontSize: '12px',
};

const childItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  padding: '5px 8px',
  borderRadius: '6px',
  color: 'var(--text, #1d1d1f)',
  cursor: 'pointer',
  fontSize: '12px',
};
