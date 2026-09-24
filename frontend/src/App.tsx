import React, { useEffect, useState } from 'react';
import { useModel } from './hooks/useModel';
import { useLayerTensors } from './hooks/useLayerTensors';
import { AttentionHeatmap } from './components/visualizations/panels/AttentionHeatmap';
import { ActivationHeatmap } from './components/visualizations/panels/ActivationHeatmap';
import { TokenViewer } from './components/visualizations/panels/TokenViewer';
import { NeuronUMAP } from './components/visualizations/neuron-umap/NeuronUMAP';
import { buildNeuronPoints, idForHeadNeuron, parseNeuronId } from './components/visualizations/neuron-umap/data';
import { LayerSidebar } from './components/LayerSidebar';
import { NeuronPanel } from './components/NeuronPanel';
import { StatusBar } from './components/StatusBar';
import { ErrorUI } from './components/ErrorUI';
import { CommandPalette } from './components/CommandPalette';
import { DockManager } from './layout/DockManager';
import { TokenPanel } from './panels/TokenPanel';
import { LayerPanel } from './panels/LayerPanel';
import { PredictionPanel } from './panels/PredictionPanel';
import { SocietyPanel } from './panels/SocietyPanel';
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
import { darkColors } from './design/tokens/colors';
import './design/styles/global.css';

import CircuitExplorerView from './components/CircuitExplorerView';
import KnowledgeGraphView from './components/KnowledgeGraphView';
import BenchmarkDashboard from './components/BenchmarkDashboard';
import BenchmarkSuiteView from './components/BenchmarkSuiteView';
import ExperimentsView from './components/ExperimentsView';
import SessionsView from './components/SessionsView';
import ReportsView from './components/ReportsView';
import ModelsView from './components/ModelsView';
import Gpt2View from './components/Gpt2View';
import Gpt2NeuronExplorer from './components/Gpt2NeuronExplorer';
import TransformerVisualizer from './components/TransformerVisualizer';
import { TransformerExplorer } from './components/visualizations/TransformerExplorer';
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
  gpt2explorer: { label: 'GPT-2 Neuron Explorer' },
  transformer: { label: 'Transformer Visualizer' },
  transformerExplorer: { label: 'Transformer Explorer' },
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
  const { state: model, listModels, load, infer, clearError } = useModel();
  const [appState, setAppState] = useAppStore();

  const [panel, setPanel] = useState<PanelState>({
    selectedLayer: 0, selectedHead: 0, selectedNeuron: null,
    hoveredToken: null, error: null,
  });

  const [activityCollapsed, setActivityCollapsed] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => { listModels(); }, []);

  const setError = (e: string | null) => setPanel(s => ({ ...s, error: e }));

  const setActivePage = (page: string) => {
    setAppState({ activePage: page });
    if (typeof window !== 'undefined' && window.location.hash.slice(1) !== page) {
      window.location.hash = page;
    }
  };

  // Hash-router: activePage stays the source of truth; the URL hash mirrors it
  // so every page is addressable by route and back/forward works.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const current = window.location.hash.slice(1);
    if (current && PAGES[current]) {
      setAppState({ activePage: current });
    }
    const onHashChange = () => {
      const next = window.location.hash.slice(1);
      if (next && PAGES[next]) setAppState({ activePage: next });
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const [showDiscoveryMemory, setShowDiscoveryMemory] = useState(false);
  const [showShare, setShowShare] = useState(false);
  const [showMarketplace, setShowMarketplace] = useState(false);
  const [showExport, setShowExport] = useState(false);

  const data = model.result;
  const layer = data?.layers[panel.selectedLayer];
  const head = layer?.heads[panel.selectedHead];
  const { tensors: layerTensors, loading: tensorsLoading } = useLayerTensors(
    data ? prompt : null,
    panel.selectedLayer,
  );

  /** Per-token spectra for the top-16 bar neurons, from live mlp_post. */
  const neuronSpectra = React.useMemo(() => {
    if (!layerTensors || !head) return head?.neurons.map(() => null) ?? [];
    return head.neurons.map(n => {
      const col = layerTensors.mlp_post.map(row => row[n.index] ?? 0);
      return col.length > 0 ? col : null;
    });
  }, [layerTensors, head]);

  /** Per-token residual L2 norms for the selected layer. */
  const residNorms = React.useMemo(() => {
    if (!layerTensors) return null;
    return layerTensors.resid_post.map(row =>
      Math.sqrt(row.reduce((s, v) => s + v * v, 0)));
  }, [layerTensors]);

  /** Mean row-entropy of the selected head's live attention matrix. */
  const headEntropy = React.useMemo(() => {
    if (!head) return null;
    const rows = head.attentionMatrix;
    if (!rows.length) return null;
    let sum = 0;
    let count = 0;
    for (const row of rows) {
      const tot = row.reduce((s, v) => s + v, 0) || 1;
      for (const v of row) {
        const q = v / tot;
        if (q > 0) sum -= q * Math.log(q);
      }
      count++;
    }
    return count ? sum / count : null;
  }, [head]);
  const umapPoints = React.useMemo(
    () => (data ? buildNeuronPoints(data.layers, data.tokens.map(t => t.text)) : []),
    [data],
  );

  const [prompt, setPrompt] = useState('Hello world');

  const handleRun = () => { if (prompt.trim()) infer(prompt); };
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleRun(); }
  };

  const handleLoadModel = (name: string) => {
    load(name);
    setPanel(s => ({ ...s, selectedLayer: 0, selectedHead: 0, selectedNeuron: null }));
  };

  const pageProps = { api, onNavigate: setActivePage };

  const renderPage = () => {
    switch (appState.activePage) {
      case 'explorer':
        return renderExplorer();
      case 'gpt2':
        return <Gpt2View {...pageProps} />;
      case 'gpt2explorer':
        return <Gpt2NeuronExplorer {...pageProps} />;
      case 'transformer':
        return <TransformerVisualizer {...pageProps} />;
      case 'transformerExplorer':
        // Needs model data (tokens/layers) — the old {...pageProps} spread
        // passed api/onNavigate instead and crashed buildNeuronPoints.
        return data?.tokens && data?.layers?.length ? (
          <TransformerExplorer
            tokens={data.tokens.map(t => t.text)}
            layers={data.layers}
            numLayers={data.layers.length}
            numHeads={data.layers[0]?.heads.length ?? 12}
          />
        ) : (
          <div className="welcome-screen">
            <div className="welcome-hint">Load a model and run a prompt to populate transformer data.</div>
          </div>
        );
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
              <>
                <div className="model-selector">
                  {model.availableModels.map(name => (
                    <button key={name} onClick={() => handleLoadModel(name)} className="btn model-load-btn">
                      Load {name}
                    </button>
                  ))}
                </div>
                <div className="dock-panel">
                  <div className="dock-panel-header">
                    <span>Society Runs</span>
                  </div>
                  <div className="dock-panel-body">
                    <SocietyPanel />
                  </div>
                </div>
              </>
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
              <>
                <div className="inference-empty">
                  {model.running ? 'Running inference...' : 'Enter a prompt and click Run'}
                </div>
                <div className="dock-panel">
                  <div className="dock-panel-header">
                    <span>Society Runs</span>
                  </div>
                  <div className="dock-panel-body">
                    <SocietyPanel />
                  </div>
                </div>
              </>
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
                  <DockManager>
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
                          tokens={layerTensors?.tokens ?? data.tokens.map(t => t.text)}
                          neuronTokenActivations={neuronSpectra}
                        />
                      ) : <div className="hint">Select a layer and head</div>,
                      neuron_panel: (
                        <NeuronPanel
                          neurons={head?.neurons ?? []}
                          selectedNeuron={panel.selectedNeuron}
                          onSelectNeuron={n => setPanel(s => ({ ...s, selectedNeuron: n }))}
                          tokens={data?.tokens.map(t => t.text) ?? []}
                        />
                      ),
                      neuron_umap: (
                        <NeuronUMAP
                          points={umapPoints}
                          tokens={data.tokens.map(t => t.text)}
                          selectedId={panel.selectedNeuron !== null ? idForHeadNeuron(panel.selectedLayer, panel.selectedHead, panel.selectedNeuron) : null}
                          onSelectNeuron={id => {
                            const parsed = parseNeuronId(id);
                            setPanel(s => parsed
                              ? { ...s, selectedLayer: parsed.layer, selectedHead: parsed.head ?? s.selectedHead, selectedNeuron: parsed.neuron }
                              : { ...s, selectedNeuron: null });
                          }}
                        />
                      ),
                      token_inspector: (
                        <TokenPanel
                          tokens={data.tokens.map(t => t.text)}
                          tokenIds={data.tokens.map(t => t.id)}
                          selectedTokenIdx={appState.selection.selectedTokenIdx}
                          onSelectToken={idx => setAppState(prev => ({ selection: { ...prev.selection, selectedTokenIdx: idx } }))}
                          residNorms={residNorms}
                        />
                      ),
                      layer_inspector: (
                        <LayerPanel
                          layerIdx={panel.selectedLayer}
                          numHeads={data.layers[panel.selectedLayer]?.heads.length ?? 12}
                          residL2={layerTensors?.stats.resid_l2 ?? null}
                          mlpMean={layerTensors?.stats.mlp_mean ?? null}
                          mlpSparsity={layerTensors?.stats.mlp_sparsity ?? null}
                          headEntropy={headEntropy}
                          loading={tensorsLoading}
                        />
                      ),
                      prediction_inspector: (
                        <PredictionPanel
                          prompt={prompt}
                        />
                      ),
                      society: <SocietyPanel />,
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
    <div className={`app${sidebarCollapsed ? ' sidebar-collapsed' : ''}${activityCollapsed ? ' activity-collapsed' : ''}`}>
      <ActivityBar
        active={appState.activePage}
        onSelect={setActivePage}
        collapsed={activityCollapsed}
        onToggle={() => setActivityCollapsed(!activityCollapsed)}
        onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <Sidebar
        pages={PAGES}
        active={appState.activePage}
        onSelect={setActivePage}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
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
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
          activityCollapsed={activityCollapsed}
          onToggleActivity={() => setActivityCollapsed(!activityCollapsed)}
        />
        <div className="content">
          <CommandPalette
            onLoadModel={handleLoadModel}
            onRunPrompt={handleRun}
          />
          {renderPage()}
        </div>
        <ConsolePanel logs={[]} />
        <DiscoveryMemoryModal isOpen={showDiscoveryMemory} onClose={() => setShowDiscoveryMemory(false)} />
        <WorkspaceSharingModal isOpen={showShare} onClose={() => setShowShare(false)} />
        <ExtensionMarketplaceModal isOpen={showMarketplace} onClose={() => setShowMarketplace(false)} />
        <PublicationExportModal isOpen={showExport} onClose={() => setShowExport(false)} />

        <ErrorUI message={panel.error || model.error} onDismiss={() => { setError(null); clearError(); }} />
      </div>
      <StatusBar
        modelName={model.modelInfo?.model_name ?? 'GPT2'}
        gpuUtil={data?.gpuUtil ?? 0}
        memoryUtil={data?.memoryUtil ?? 0}
        tokenCount={data?.tokens.length ?? 0}
      />
    </div>
  );
}
