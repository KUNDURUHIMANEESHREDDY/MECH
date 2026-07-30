import React, { useEffect, useState, useCallback } from 'react';
import Sidebar from './components/Sidebar.jsx';
import Topbar from './components/Topbar.jsx';
import Workspace from './components/Workspace.jsx';
import ModelsView from './components/ModelsView.jsx';
import PromptsView from './components/PromptsView.jsx';
import DebuggerView from './components/DebuggerView.jsx';
import Gpt2View from './components/Gpt2View.jsx';
import ExperimentsView from './components/ExperimentsView.jsx';
import SessionsView from './components/SessionsView.jsx';
import ReportsView from './components/ReportsView.jsx';
import Settings from './components/Settings.jsx';
import Logging from './components/Logging.jsx';
import BuildLog from './components/BuildLog.jsx';
import CommandPalette from './components/CommandPalette.jsx';
import DockManager from './components/DockManager.jsx';
import WorkspaceSharingModal from './components/WorkspaceSharingModal.jsx';
import ExtensionMarketplaceModal from './components/ExtensionMarketplaceModal.jsx';
import PublicationExportModal from './components/PublicationExportModal.jsx';
import NeuralExplorerView from './components/NeuralExplorerView.jsx';
import BenchmarkDashboard from './components/BenchmarkDashboard.jsx';
import KnowledgeGraphView from './components/KnowledgeGraphView.jsx';
import CircuitExplorerView from './components/CircuitExplorerView.jsx';

import ReasoningTraceView from './components/ReasoningTraceView.jsx';
import EvidenceFusionView from './components/EvidenceFusionView.jsx';
import DiscoveryMemoryModal from './components/DiscoveryMemoryModal.jsx';
import CampaignWorkspaceView from './components/CampaignWorkspaceView.jsx';
import ResearchAnalyticsView from './components/ResearchAnalyticsView.jsx';
import ScientificHealthView from './components/ScientificHealthView.jsx';
import PluginSDKView from './components/PluginSDKView.jsx';
import BenchmarkSuiteView from './components/BenchmarkSuiteView.jsx';

const PAGES = {
  workspace:  { label: 'Workspace',    component: Workspace,    crumb: 'Workspace' },
  gpt2:       { label: 'GPT-2 Live',   component: Gpt2View,     crumb: 'GPT-2 Small — Live Interpretability' },
  campaigns: { label: 'Campaigns', component: CampaignWorkspaceView, crumb: 'Research Campaigns Workspace' },
  analytics: { label: 'Analytics', component: ResearchAnalyticsView, crumb: 'Multi-Campaign Meta-Learning Analytics' },
  models: { label: 'Models', component: ModelsView, crumb: 'Models Explorer' },
  prompts: { label: 'Prompts', component: PromptsView, crumb: 'Prompt Library' },
  debugger: { label: 'Debugger', component: DebuggerView, crumb: 'Circuit Debugger' },
  experiments: { label: 'Experiments', component: ExperimentsView, crumb: 'Experiments Matrix' },
  sessions: { label: 'Sessions', component: SessionsView, crumb: 'Session Explorer' },
  reports: { label: 'Reports', component: ReportsView, crumb: 'Research Reports' },
  settings: { label: 'Settings', component: Settings, crumb: 'Settings' },
  logging: { label: 'Logging', component: Logging, crumb: 'Logging' },
  build:          { label: 'Build',           component: BuildLog,          crumb: 'Build' },
  neuralexplorer: { label: 'Neural Explorer', component: NeuralExplorerView, crumb: 'Interactive Neural Explorer' },
  benchmark: { label: 'Benchmarks', component: BenchmarkDashboard, crumb: 'Reproducibility Benchmarks' },
  knowledgegraph: { label: 'Knowledge Graph', component: KnowledgeGraphView, crumb: 'Mechanistic Knowledge Graph' },
  circuitexplorer: { label: 'Circuit Explorer', component: CircuitExplorerView, crumb: 'Circuit Explorer' },
  reasoning: { label: 'Scientific Reasoning', component: ReasoningTraceView, crumb: 'Reasoning Trace Viewer' },
  evidencefusion: { label: 'Evidence Fusion', component: EvidenceFusionView, crumb: 'Evidence Fusion Engine' },
  health: { label: 'Scientific Health', component: ScientificHealthView, crumb: 'Continuous Validation & Regression Monitoring' },
  plugins: { label: 'Plugin SDK', component: PluginSDKView, crumb: 'Plugin SDK — Extend the Research Platform' },
  benchmarksuite: { label: 'MI Benchmarks', component: BenchmarkSuiteView, crumb: 'Real MI Benchmark Suite — 8 Tasks × 5 Models' },
};

export default function App() {
  const [active, setActive] = useState('workspace');
  const [settings, setSettings] = useState(null);
  const [pythonStatus, setPythonStatus] = useState('connecting');
  const [isCmdPaletteOpen, setIsCmdPaletteOpen] = useState(false);
  const [isDockOpen, setIsDockOpen] = useState(false);
  const [isShareOpen, setIsShareOpen] = useState(false);
  const [isMarketplaceOpen, setIsMarketplaceOpen] = useState(false);
  const [isPublicationOpen, setIsPublicationOpen] = useState(false);
  const [isDiscoveryMemoryOpen, setIsDiscoveryMemoryOpen] = useState(false);

  // Dynamic API getter resolving window.appApi
  const getApi = () => (typeof window !== 'undefined' ? window.appApi : undefined);

  // Load settings on mount and apply theme
  useEffect(() => {
    const api = getApi();
    if (!api || !api.getSettings) return;
    api.getSettings().then((s) => {
      setSettings(s);
      applyTheme(s?.theme || 'system');
    }).catch(() => undefined);
  }, []);

  // Ping Python sidecar
  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      const api = getApi();
      if (!api || !api.pythonPing) return;
      try {
        const res = await api.pythonPing();
        if (cancelled) return;
        const isConn = res && (res.ok || res.echo === 'pong');
        setPythonStatus(isConn ? 'connected' : 'error');
      } catch (err) {
        if (!cancelled) setPythonStatus('error');
      }
    };
    ping();
    const id = setInterval(ping, 3_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Global Ctrl+Shift+P shortcut
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'P' || e.key === 'p')) {
        e.preventDefault();
        setIsCmdPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleSettingsChange = useCallback(async (patch) => {
    const api = getApi();
    if (!api || !api.setSettings) return;
    const next = await api.setSettings(patch);
    setSettings(next);
    if (patch && patch.theme) applyTheme(patch.theme);
  }, []);

  const handleExecuteCommand = (cmdId) => {
    if (cmdId === 'load-model')          setActive('models');
    else if (cmdId === 'run-prompt')     setActive('gpt2');
    else if (cmdId === 'gpt2')           setActive('gpt2');
    else if (cmdId === 'open-experiments') setActive('experiments');
    else if (cmdId === 'open-sessions')  setActive('sessions');
    else if (cmdId === 'toggle-dock')    setIsDockOpen((prev) => !prev);
    else if (cmdId === 'toggle-theme') {
      const nextTheme = settings?.theme === 'light' ? 'dark' : 'light';
      handleSettingsChange({ theme: nextTheme });
    } else {
      setActive('debugger');
    }
  };

  const Page = PAGES[active]?.component || Workspace;
  const currentApi = getApi();

  return (
    <div className="app">
      <Sidebar
        pages={PAGES}
        active={active}
        onSelect={setActive}
      />
      <div className="main">
        <Topbar
          crumb={PAGES[active]?.crumb}
          pythonStatus={pythonStatus}
          onToggleDock={() => setIsDockOpen((prev) => !prev)}
          onToggleCmdPalette={() => setIsCmdPaletteOpen(true)}
          onOpenShare={() => setIsShareOpen(true)}
          onOpenMarketplace={() => setIsMarketplaceOpen(true)}
          onOpenPublication={() => setIsPublicationOpen(true)}
          onOpenDiscoveryMemory={() => setIsDiscoveryMemoryOpen(true)}
        />
        <div className="content" data-testid="content">
          {active === 'settings' ? (
            <Page
              settings={settings}
              onChange={handleSettingsChange}
              api={currentApi}
              onNavigate={setActive}
            />
          ) : (
            <Page api={currentApi} settings={settings} onNavigate={setActive} />
          )}
        </div>

        <DockManager
          isVisible={isDockOpen}
          onClose={() => setIsDockOpen(false)}
          activeTab={active}
          setActiveTab={setActive}
        />
      </div>

      <CommandPalette
        isOpen={isCmdPaletteOpen}
        onClose={() => setIsCmdPaletteOpen(false)}
        onExecuteCommand={handleExecuteCommand}
      />

      <WorkspaceSharingModal
        isOpen={isShareOpen}
        onClose={() => setIsShareOpen(false)}
      />

      <ExtensionMarketplaceModal
        isOpen={isMarketplaceOpen}
        onClose={() => setIsMarketplaceOpen(false)}
      />

      <PublicationExportModal
        isOpen={isPublicationOpen}
        onClose={() => setIsPublicationOpen(false)}
        figureData={{ title: "GPT-2 Mechanistic Analysis" }}
      />

      <DiscoveryMemoryModal
        isOpen={isDiscoveryMemoryOpen}
        onClose={() => setIsDiscoveryMemoryOpen(false)}
        onNavigate={setActive}
      />
    </div>
  );
}

function applyTheme(theme) {
  if (typeof document === 'undefined') return;
  const root = document.documentElement;
  if (theme === 'light') {
    root.setAttribute('data-theme', 'light');
  } else if (theme === 'dark') {
    root.setAttribute('data-theme', 'dark');
  } else {
    root.removeAttribute('data-theme');
  }
}
