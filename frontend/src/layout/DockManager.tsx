import React from 'react';
import { useWorkspaceStore } from '../shared/stores/workspace';
import { pluginRegistry } from '../panel-system/pluginRegistry';

import { Gpt2View } from '../components/Gpt2View';
import { ModelsView } from '../components/ModelsView';
import { TransformerVisualizer } from '../components/TransformerVisualizer';
import { NeuralExplorerView } from '../components/NeuralExplorerView';
import { CircuitExplorerView } from '../components/CircuitExplorerView';
import { KnowledgeGraphView } from '../components/KnowledgeGraphView';
import { DebuggerView } from '../components/DebuggerView';
import { BenchmarkDashboard } from '../components/BenchmarkDashboard';
import { BenchmarkSuiteView } from '../components/BenchmarkSuiteView';
import { ExperimentsView } from '../components/ExperimentsView';
import { ReasoningTraceView } from '../components/ReasoningTraceView';
import { EvidenceFusionView } from '../components/EvidenceFusionView';
import { ResearchAnalyticsView } from '../components/ResearchAnalyticsView';
import { ScientificHealthView } from '../components/ScientificHealthView';
import { CampaignWorkspaceView } from '../components/CampaignWorkspaceView';
import { Workspace } from '../components/Workspace';
import { PromptsView } from '../components/PromptsView';
import { BuildLog } from '../components/BuildLog';
import { SessionsView } from '../components/SessionsView';
import { ReportsView } from '../components/ReportsView';
import { PluginSDKView } from '../components/PluginSDKView';
import { ResearchNotebook } from '../components/ResearchNotebook';
import { PaperReproductionView } from '../components/PaperReproductionView';
import { Settings } from '../components/Settings';
import { Logging } from '../components/Logging';
import { Projects } from '../components/Projects';
import { RecentFiles } from '../components/RecentFiles';
import { Gpt2NeuronExplorer } from '../components/Gpt2NeuronExplorer';

const VIEW_COMPONENT_MAP: Record<string, { title: string; Component: React.ComponentType<any>; fullWidth?: boolean }> = {
  gpt2: { title: 'GPT-2 Live Engine', Component: Gpt2View, fullWidth: true },
  explorer: { title: 'Model Explorer', Component: ModelsView, fullWidth: true },
  models: { title: 'Models Catalog', Component: ModelsView, fullWidth: true },
  transformer: { title: 'Transformer Visualizer', Component: TransformerVisualizer, fullWidth: true },
  neuralexplorer: { title: 'Neural Explorer', Component: NeuralExplorerView, fullWidth: true },
  circuitexplorer: { title: 'Circuit Explorer', Component: CircuitExplorerView, fullWidth: true },
  knowledgegraph: { title: 'Knowledge Graph', Component: KnowledgeGraphView, fullWidth: true },
  debugger: { title: 'Neural Debugger', Component: DebuggerView, fullWidth: true },
  benchmark: { title: 'Benchmark Dashboard', Component: BenchmarkDashboard, fullWidth: true },
  benchmarksuite: { title: 'Benchmark Suite', Component: BenchmarkSuiteView, fullWidth: true },
  experiments: { title: 'Experiments', Component: ExperimentsView, fullWidth: true },
  reasoning: { title: 'Reasoning Trace', Component: ReasoningTraceView, fullWidth: true },
  evidencefusion: { title: 'Evidence Fusion', Component: EvidenceFusionView, fullWidth: true },
  analytics: { title: 'Research Analytics', Component: ResearchAnalyticsView, fullWidth: true },
  health: { title: 'Scientific Health', Component: ScientificHealthView, fullWidth: true },
  campaigns: { title: 'Campaign Workspace', Component: CampaignWorkspaceView, fullWidth: true },
  workspace: { title: 'Workspace Dashboard', Component: Workspace, fullWidth: true },
  prompts: { title: 'Prompts Manager', Component: PromptsView, fullWidth: true },
  build: { title: 'Build Output', Component: BuildLog, fullWidth: true },
  sessions: { title: 'Sessions Manager', Component: SessionsView, fullWidth: true },
  reports: { title: 'Reports & Papers', Component: ReportsView, fullWidth: true },
  plugins: { title: 'Plugins & SDK', Component: PluginSDKView, fullWidth: true },
  notebook: { title: 'Research Notebook', Component: ResearchNotebook, fullWidth: true },
  labnotebook: { title: 'Lab Notebook', Component: ResearchNotebook, fullWidth: true },
  reproduction: { title: 'Paper Reproduction', Component: PaperReproductionView, fullWidth: true },
  settings: { title: 'Settings', Component: Settings, fullWidth: true },
  logging: { title: 'System Logging', Component: Logging, fullWidth: true },
  projects: { title: 'Projects Manager', Component: Projects, fullWidth: true },
  recent: { title: 'Recent Files', Component: RecentFiles, fullWidth: true },
  gpt2explorer: { title: 'GPT-2 Neuron Explorer', Component: Gpt2NeuronExplorer, fullWidth: true },
};

const EmptyState: React.FC = () => (
  <div
    style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'var(--text-muted, #7a7a7a)',
      gap: '16px',
      fontSize: '14px',
      userSelect: 'none',
      padding: '40px',
    }}
  >
    <svg
      width="52"
      height="52"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.25"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ opacity: 0.3 }}
    >
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M9 3v18" />
      <path d="M3 9h6" />
      <path d="M3 15h6" />
    </svg>
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontWeight: 600, fontSize: '15px', marginBottom: '6px', color: 'var(--text, #1d1d1f)' }}>
        Visual Research Canvas
      </div>
      <div style={{ fontSize: '13px', color: 'var(--text-muted, #7a7a7a)', maxWidth: '280px', lineHeight: 1.5 }}>
        Click a resource in the Navigator or press <kbd style={{ background: 'var(--bg-elev-2, #f0f0f0)', border: '1px solid var(--border, #d0d0d0)', borderRadius: '4px', padding: '1px 5px', fontSize: '11px', fontWeight: 600 }}>⌘K</kbd> to open a view.
      </div>
    </div>
  </div>
);

export const DockManager: React.FC = () => {
  const visiblePanels = useWorkspaceStore((s) => s.visiblePanels);
  const closePanel = useWorkspaceStore((s) => s.closePanel);

  const openPanelIds = Object.entries(visiblePanels)
    .filter(([, visible]) => visible)
    .map(([id]) => id);

  if (openPanelIds.length === 0) {
    return <EmptyState />;
  }

  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexWrap: 'wrap',
        gap: '8px',
        padding: '12px',
        overflow: 'auto',
        alignContent: 'flex-start',
        height: '100%',
        boxSizing: 'border-box',
      }}
    >
      {openPanelIds.map((panelId) => {
        const plugin = pluginRegistry.get(panelId);
        const viewEntry = VIEW_COMPONENT_MAP[panelId];

        if (!plugin && !viewEntry) return null;

        const title = plugin ? plugin.title : viewEntry.title;
        const fullWidth = viewEntry?.fullWidth;

        const dummyCtx: any = {
          selection: { resource: null, layer: null, head: null, neuron: null, token: null },
          workspace: {},
        };

        return (
          <div
            key={panelId}
            style={{
              background: 'var(--color-canvas, #ffffff)',
              border: '1px solid var(--border, #e0e0e0)',
              borderRadius: '10px',
              minWidth: fullWidth ? '100%' : '320px',
              flex: fullWidth ? '1 1 100%' : '1 1 320px',
              maxWidth: fullWidth ? '100%' : '600px',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
              marginBottom: '8px',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 12px',
                borderBottom: '1px solid var(--border-light, #f0f0f0)',
                background: 'var(--bg-elev-1, #fafafa)',
                fontSize: '12px',
                fontWeight: 600,
                color: 'var(--text, #1d1d1f)',
              }}
            >
              <span>{title}</span>
              <button
                onClick={() => closePanel(panelId)}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-muted, #7a7a7a)',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  fontSize: '16px',
                  lineHeight: 1,
                }}
                title="Close panel"
              >
                ×
              </button>
            </div>
            <div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>
              {plugin ? (
                <plugin.Body {...dummyCtx} />
              ) : (
                <viewEntry.Component />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};
