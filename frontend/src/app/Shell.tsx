import React, { useState, useEffect } from 'react';
import { Toolbar } from '../shell/toolbar/Toolbar';
import { Navigator } from '../shell/navigator/Navigator';
import { WorkspaceCanvas } from '../shell/workspace/WorkspaceCanvas';
import { Inspector } from '../shell/inspector/Inspector';
import { BottomWorkspace } from '../shell/bottom/BottomWorkspace';
import { CommandPalette } from '../shell/toolbar/CommandPalette';
import { DockManager } from '../layout/DockManager';
import { useUIStore } from '../shared/stores/ui';

interface ShellProps {
  children?: React.ReactNode;
  pythonStatus?: boolean;
  onOpenCommandPalette?: () => void;
  onLoadModel?: (name: string) => void;
  onRunPrompt?: () => void;
}

const PAGE_LABEL_MAP: Record<string, string> = {
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
  build: 'Build',
  benchmark: 'Benchmark',
  benchmarksuite: 'Benchmark Suite',
  experiments: 'Experiments',
  sessions: 'Sessions',
  reports: 'Reports',
  reasoning: 'Reasoning',
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
  gpt2explorer: 'GPT-2 Neuron Explorer',
  transformerExplorer: 'Transformer Explorer',
};

export const Shell: React.FC<ShellProps> = ({
  children,
  pythonStatus = true,
  onOpenCommandPalette,
  onLoadModel,
  onRunPrompt,
}) => {
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const activityCollapsed = useUIStore((s) => s.activityCollapsed);
  const sidebarCollapsed = useUIStore((s) => s.sidebarCollapsed);
  const setCrumb = useUIStore((s) => s.setCrumb);
  const openPanel = useWorkspaceStore((s) => s.openPanel);

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace(/^#/, '');
      if (hash && PAGE_LABEL_MAP[hash]) {
        setCrumb(PAGE_LABEL_MAP[hash]);
        openPanel(hash);
      }
    };
    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [setCrumb, openPanel]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      }
      if (e.key === 'Escape' && commandPaletteOpen) {
        setCommandPaletteOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [commandPaletteOpen]);

  const handleOpenCommandPalette = () => {
    setCommandPaletteOpen(true);
  };

  const handleCommandPaletteOpenChange = (open: boolean) => {
    setCommandPaletteOpen(open);
  };

  return (
    <div
      className={`shell-app-container app ${activityCollapsed ? 'activity-collapsed' : ''}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        background: 'var(--color-canvas-parchment, #f5f5f7)',
        color: 'var(--text, #1d1d1f)',
        fontFamily: 'var(--font, system-ui, sans-serif)',
      }}
    >
      <Toolbar onOpenCommandPalette={handleOpenCommandPalette} pythonStatus={pythonStatus} />
      <div style={{ flex: 1, display: 'flex', minHeight: 0, overflow: 'hidden' }}>
        <Navigator />
        <WorkspaceCanvas>
          <DockManager />
        </WorkspaceCanvas>
          <Inspector />
      </div>
      <BottomWorkspace />
      <CommandPalette open={commandPaletteOpen} onOpenChange={handleCommandPaletteOpenChange} />
    </div>
  );
};
