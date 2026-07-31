import React, { useEffect, useState } from 'react';
import { useModel } from './hooks/useModel';
import { AttentionHeatmap } from './components/visualizations/panels/AttentionHeatmap';
import { ActivationHeatmap } from './components/visualizations/panels/ActivationHeatmap';
import { TokenViewer } from './components/visualizations/panels/TokenViewer';
import { LayerSidebar } from './components/LayerSidebar';
import { NeuronPanel } from './components/NeuronPanel';
import { StatusBar } from './components/StatusBar';
import { ErrorUI } from './components/ErrorUI';
import { CommandPalette } from './components/CommandPalette';
import { DockManager } from './layout/DockManager';
import { TokenPanel } from './panels/TokenPanel';
import { LayerPanel } from './panels/LayerPanel';
import { PredictionPanel } from './panels/PredictionPanel';
import { useAppStore } from './store/useAppStore';
import { PanelState } from './types';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import ActivityBar from './components/ActivityBar';
import ConsolePanel from './components/ConsolePanel';
import { api } from './services/api';
import DiscoveryMemoryModal from './components/DiscoveryMemoryModal';
import WorkspaceSharingModal from './components/WorkspaceSharingModal';
import ExtensionMarketplaceModal from './components/ExtensionMarketplaceModal';
import PublicationExportModal from './components/PublicationExportModal';

import CircuitExplorerView from './components/CircuitExplorerView';
import KnowledgeGraphView from './components/KnowledgeGraphView';
import BenchmarkDashboard from './components/BenchmarkDashboard';
import BenchmarkSuiteView from './components/BenchmarkSuiteView';
import ExperimentsView from './components/ExperimentsView';
import SessionsView from './components/SessionsView';
import ReportsView from './components/ReportsView';
import ModelsView from './components/ModelsView';
import Gpt2View from './components/Gpt2View';
import DebuggerView from './components/DebuggerView';
import PromptsView from './components/PromptsView';
import NeuralExplorerView from './components/NeuralExplorerView';
import ResearchAnalyticsView from './components/ResearchAnalyticsView';
import ScientificHealthView from './components/ScientificHealthView';
import PaperReproductionView from './components/PaperReproductionView';
import PluginSDKView from './components/PluginSDKView';
import ReasoningTraceView from './components/ReasoningTraceView';
import EvidenceFusionView from './components/EvidenceFusionView';
import ResearchNotebook from './components/ResearchNotebook';
import ExperimentNotebook from './components/ExperimentNotebook';
import Settings from './components/Settings';
import Logging from './components/Logging';
import BuildLog from './components/BuildLog';
import Projects from './components/Projects';
import RecentFiles from './components/RecentFiles';
import CampaignWorkspaceView from './components/CampaignWorkspaceView';

const PAGES: Record<string, { label: string }> = {
  explorer: { label: 'Model Explorer' },
  gpt2: { label: 'GPT-2 Live' },
  workspace: { label: 'Workspace' },
  models: { label: 'Models' },
  prompts: { label: 'Prompts' },
  debugger: { label: 'Debugger' },
  experiments: { label: 'Experiments' },
  sessions: { label: 'Sessions' },
  reports: { label: 'Reports' },
  settings: { label: 'Settings' },
  logging: { label: 'Logging' },
  build: { label: 'Build' },
  neuralexplorer: { label: 'Neural Explorer' },
  benchmark: { label: 'Benchmark' },
  benchmarksuite: { label: 'Benchmark Suite' },
  knowledgegraph: { label: 'Knowledge Graph' },
  circuitexplorer: { label: 'Circuit Explorer' },
  reasoning: { label: 'Reasoning' },
  evidencefusion: { label: 'Evidence Fusion' },
  campaigns: { label: 'Campaigns' },
  analytics: { label: 'Analytics' },
  health: { label: 'Health' },
  plugins: { label: 'Plugins' },
  notebook: { label: 'Research Notebook' },
  labnotebook: { label: 'Lab Notebook' },
  reproduction: { label: 'Paper Reproduction' },
  projects: { label: 'Projects' },
  recent: { label: 'Recent Files' },
};

export default function App() {
  const { state: model, listModels, load, infer } = useModel();
  const [appState, setAppState] = useAppStore();

  const [panel, setPanel] = useState<PanelState>({
    selectedLayer: 0, selectedHead: 0, selectedNeuron: null,
    hoveredToken: null, error: null, darkMode: appState.darkMode,
  });

  useEffect(() => { listModels(); }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', appState.darkMode ? 'dark' : 'light');
  }, [appState.darkMode]);

  const setError = (e: string | null) => setPanel(s => ({ ...s, error: e }));
  const toggleDarkMode = () => {
    const nextDark = !appState.darkMode;
    setAppState({ darkMode: nextDark });
    setPanel(s => ({ ...s, darkMode: nextDark }));
  };

  const setActivePage = (page: string) => setAppState({ activePage: page });

  const [showDiscoveryMemory, setShowDiscoveryMemory] = useState(false);
  const [showShare, setShowShare] = useState(false);
  const [showMarketplace, setShowMarketplace] = useState(false);
  const [showExport, setShowExport] = useState(false);

  const data = model.result;
  const layer = data?.layers[panel.selectedLayer];
  const head = layer?.heads[panel.selectedHead];

  const [prompt, setPrompt] = useState('Hello world');

  const handleRun = () => { if (prompt.trim()) infer(prompt); };
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleRun(); }
  };

  const handleLoadModel = (name: string) => {
    load(name);
    setPanel(s => ({ ...s, selectedLayer: 0, selectedHead: 0, selectedNeuron: null }));
  };

  const pageProps = { api, onNavigate: setActivePage, darkMode: appState.darkMode };

  const renderPage = () => {
    switch (appState.activePage) {
      case 'explorer':
        return renderExplorer();
      case 'gpt2':
        return <Gpt2View {...pageProps} />;
      case 'workspace':
        return <CampaignWorkspaceView {...pageProps} />;
      case 'models':
        return <ModelsView {...pageProps} />;
      case 'prompts':
        return <PromptsView {...pageProps} />;
      case 'debugger':
        return <DebuggerView {...pageProps} />;
      case 'experiments':
        return <ExperimentsView {...pageProps} />;
      case 'sessions':
        return <SessionsView {...pageProps} />;
      case 'reports':
        return <ReportsView />;
      case 'settings':
        return <Settings settings={{}} onChange={() => {}} api={api} />;
      case 'logging':
        return <Logging {...pageProps} />;
      case 'build':
        return <BuildLog {...pageProps} />;
      case 'neuralexplorer':
        return <NeuralExplorerView {...pageProps} />;
      case 'benchmark':
        return <BenchmarkDashboard />;
      case 'benchmarksuite':
        return <BenchmarkSuiteView />;
      case 'knowledgegraph':
        return <KnowledgeGraphView {...pageProps} />;
      case 'circuitexplorer':
        return <CircuitExplorerView {...pageProps} />;
      case 'reasoning':
        return <ReasoningTraceView {...pageProps} />;
      case 'evidencefusion':
        return <EvidenceFusionView {...pageProps} />;
      case 'campaigns':
        return <CampaignWorkspaceView {...pageProps} />;
      case 'analytics':
        return <ResearchAnalyticsView {...pageProps} />;
      case 'health':
        return <ScientificHealthView {...pageProps} />;
      case 'plugins':
        return <PluginSDKView />;
      case 'notebook':
        return <ResearchNotebook />;
      case 'labnotebook':
        return <ExperimentNotebook />;
      case 'reproduction':
        return <PaperReproductionView />;
      case 'projects':
        return <Projects {...pageProps} />;
      case 'recent':
        return <RecentFiles {...pageProps} />;
      default:
        return (
          <div className="welcome-screen">
            <div className="welcome-hint">{PAGES[appState.activePage]?.label ?? 'Page'}</div>
            <div className="welcome-hint" style={{ fontSize: 12, opacity: 0.6 }}>Coming soon</div>
          </div>
        );
    }
  };

  const renderExplorer = () => {
    return (
      <>
        {!model.loaded && !model.loading && (
          <div className="welcome-screen">
            <div className="welcome-hint">Choose a model to load</div>
            {model.availableModels.length === 0 && model.error ? (
              <div className="error-text">Cannot reach runtime at localhost:8000. Start the backend first.</div>
            ) : (
              <div className="model-selector">
                {model.availableModels.map(name => (
                  <button key={name} onClick={() => handleLoadModel(name)} className="btn model-load-btn">
                    Load {name}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {model.loading && (
          <div className="welcome-screen">
            <div className="welcome-hint">Loading {model.modelInfo?.model_name ?? 'model'}...</div>
            <div className="progress-bar">
              <div className="progress-fill" />
            </div>
          </div>
        )}

        {model.loaded && (
          <>
            <div className="prompt-row">
              <input
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter prompt..."
                className="input-text prompt-input"
              />
              <button onClick={handleRun} disabled={model.running} className="btn">
                {model.running ? 'Running...' : 'Run'}
              </button>
            </div>

            {!data ? (
              <div className="inference-empty">
                {model.running ? 'Running inference...' : 'Enter a prompt and click Run'}
              </div>
            ) : (
              <div className="inference-body">
                <div className="layer-sidebar-wrapper">
                  <LayerSidebar
                    layers={data.layers}
                    selectedLayer={panel.selectedLayer}
                    selectedHead={panel.selectedHead}
                    onSelectLayer={l => setPanel(s => ({ ...s, selectedLayer: l, selectedHead: 0, selectedNeuron: null }))}
                    onSelectHead={h => setPanel(s => ({ ...s, selectedHead: h, selectedNeuron: null }))}
                  />
                </div>
                <div className="dock-wrapper">
                  <DockManager darkMode={appState.darkMode}>
                    {{
                      token_viewer: (
                        <TokenViewer
                          tokens={data.tokens.map(t => t.text)}
                          tokenIds={data.tokens.map(t => t.id)}
                          selectedToken={panel.hoveredToken}
                          onHoverToken={t => setPanel(s => ({ ...s, hoveredToken: t }))}
                        />
                      ),
                      attention_heatmap: head ? (
                        <AttentionHeatmap
                          matrix={head.attentionMatrix}
                          tokens={data.tokens.map(t => t.text)}
                          hoveredToken={panel.hoveredToken}
                          onHoverToken={t => setPanel(s => ({ ...s, hoveredToken: t }))}
                        />
                      ) : <div className="hint">Select a layer and head</div>,
                      activation_heatmap: head ? (
                        <ActivationHeatmap
                          activations={head.neurons.map(n => n.activation)}
                          neuronIndex={panel.selectedNeuron}
                          onSelectNeuron={n => setPanel(s => ({ ...s, selectedNeuron: n }))}
                        />
                      ) : <div className="hint">Select a layer and head</div>,
                      neuron_panel: (
                        <NeuronPanel
                          neurons={head?.neurons ?? []}
                          selectedNeuron={panel.selectedNeuron}
                          onSelectNeuron={n => setPanel(s => ({ ...s, selectedNeuron: n }))}
                        />
                      ),
                      token_inspector: (
                        <TokenPanel
                          tokens={data.tokens.map(t => t.text)}
                          selectedTokenIdx={appState.selection.selectedTokenIdx}
                          onSelectToken={idx => setAppState(prev => ({ selection: { ...prev.selection, selectedTokenIdx: idx } }))}
                          darkMode={appState.darkMode}
                        />
                      ),
                      layer_inspector: (
                        <LayerPanel
                          layerIdx={panel.selectedLayer}
                          numHeads={data.layers[panel.selectedLayer]?.heads.length ?? 12}
                          darkMode={appState.darkMode}
                        />
                      ),
                      prediction_inspector: (
                        <PredictionPanel
                          tokens={data.tokens.map(t => t.text)}
                          darkMode={appState.darkMode}
                        />
                      ),
                    }}
                  </DockManager>
                </div>
              </div>
            )}
          </>
        )}
      </>
    );
  };

  return (
    <div className="app">
      <ActivityBar active={appState.activePage} onSelect={setActivePage} />
      <Sidebar pages={PAGES} active={appState.activePage} onSelect={setActivePage} />
      <div className="main-area">
        <Topbar
          crumb={PAGES[appState.activePage]?.label ?? 'MECH'}
          pythonStatus={model.error ? 'offline' : 'connected'}
          onToggleCmdPalette={() => setAppState(prev => ({ commandPaletteOpen: !prev.commandPaletteOpen }))}
          onToggleDock={() => {}}
          onOpenShare={() => setShowShare(true)}
          onOpenMarketplace={() => setShowMarketplace(true)}
          onOpenPublication={() => setShowExport(true)}
          onOpenDiscoveryMemory={() => setShowDiscoveryMemory(true)}
        />
        <div className="content">
          <CommandPalette
            onLoadModel={handleLoadModel}
            onRunPrompt={handleRun}
            darkMode={appState.darkMode}
          />
          {renderPage()}
        </div>
        <ConsolePanel logs={[]} />
        <DiscoveryMemoryModal isOpen={showDiscoveryMemory} onClose={() => setShowDiscoveryMemory(false)} />
        <WorkspaceSharingModal isOpen={showShare} onClose={() => setShowShare(false)} />
        <ExtensionMarketplaceModal isOpen={showMarketplace} onClose={() => setShowMarketplace(false)} />
        <PublicationExportModal isOpen={showExport} onClose={() => setShowExport(false)} />

        <ErrorUI message={panel.error || model.error} onDismiss={() => { setError(null); }} />
      </div>
      <StatusBar
        modelName={model.modelInfo?.model_name ?? 'GPT2'}
        gpuUtil={data?.gpuUtil ?? 0}
        memoryUtil={data?.memoryUtil ?? 0}
        tokenCount={data?.tokens.length ?? 0}
        darkMode={appState.darkMode}
      />
    </div>
  );
}
