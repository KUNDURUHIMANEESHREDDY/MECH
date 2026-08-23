/**
 * Tool panels — one registered PanelPlugin per sidebar tool.
 *
 * The Navigator opens tools by navKey; each nav item in RESOURCE_TREE
 * expects a panel with id === navKey. The model-analysis cluster
 * (GPT-2 Live, Model Explorer, Transformer Visualizer, Neural Explorer,
 * Circuit Explorer, Knowledge Graph, Debugger, Benchmark, Benchmark Suite,
 * Models) uses the real panel implementations in components/panels; the
 * remaining tools reuse their existing view components.
 */
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext, ResourceKind } from '../../shared/types';

/* ---- Real panel components (components/panels) ---- */
import { GPT2LivePanel } from '../../components/panels/GPT2LivePanel';
import { ModelExplorerPanel } from '../../components/panels/ModelExplorerPanel';
import { TransformerNetworkPanel } from '../../components/panels/TransformerNetworkPanel';
import { NeuralExplorerPanel } from '../../components/panels/NeuralExplorerPanel';
import { Gpt2NeuronExplorerPanel } from '../../components/panels/Gpt2NeuronExplorerPanel';
import { KnowledgeGraphPanel } from '../../components/panels/KnowledgeGraphPanel';
import { DebuggerPanel } from '../../components/panels/DebuggerPanel';
import { BenchmarkPanel, BenchmarkSuitePanel } from '../../components/panels/BenchmarkPanel';
import { ModelsCatalogPanel } from '../../components/panels/ModelsCatalogPanel';
import { ModelInteractionPanel } from '../../components/panels/ModelInteractionPanel';

/* ---- Existing view components reused as tool bodies ---- */
import { ExperimentsView } from '../../components/ExperimentsView';
import { ReasoningTraceView } from '../../components/ReasoningTraceView';
import { EvidenceFusionView } from '../../components/EvidenceFusionView';
import { ResearchAnalyticsView } from '../../components/ResearchAnalyticsView';
import { ScientificHealthView } from '../../components/ScientificHealthView';
import { CampaignWorkspaceView } from '../../components/CampaignWorkspaceView';
import { Workspace } from '../../components/Workspace';
import { PromptsView } from '../../components/PromptsView';
import { BuildLog } from '../../components/BuildLog';
import { SessionsView } from '../../components/SessionsView';
import { ReportsView } from '../../components/ReportsView';
import { PluginSDKView } from '../../components/PluginSDKView';
import { ResearchNotebook } from '../../components/ResearchNotebook';
import { ExperimentNotebook } from '../../components/ExperimentNotebook';
import { PaperReproductionView } from '../../components/PaperReproductionView';
import { Settings } from '../../components/Settings';
import { Logging } from '../../components/Logging';
import { Projects } from '../../components/Projects';
import { RecentFiles } from '../../components/RecentFiles';
import { AIResearchAssistantView } from '../../components/AIResearchAssistantView';

/** Circuit Explorer body — the circuit plugin registers before this module loads. */
const circuitBody = pluginRegistry.get('circuit_explorer')?.Body ?? (() => null);

interface ToolSpec {
  id: string;
  title: string;
  kind: ResourceKind;
  Body: FC<PanelContext>;
  fullWidth?: boolean;
}

const TOOLS: ToolSpec[] = [
  /* ---- Model analysis cluster: real interactive panels ---- */
  { id: 'gpt2', title: 'GPT-2 Live', kind: 'model', Body: GPT2LivePanel },
  { id: 'interact', title: 'Model Interaction', kind: 'model', Body: ModelInteractionPanel },
  { id: 'explorer', title: 'Model Explorer', kind: 'model', Body: ModelExplorerPanel },
  { id: 'transformer', title: 'Transformer Visualizer', kind: 'model', Body: TransformerNetworkPanel },
  { id: 'neuralexplorer', title: 'Neural Explorer', kind: 'neuron', Body: NeuralExplorerPanel },
  { id: 'circuitexplorer', title: 'Circuit Explorer', kind: 'circuit', Body: circuitBody },
  { id: 'knowledgegraph', title: 'Knowledge Graph', kind: 'circuit', Body: KnowledgeGraphPanel },
  { id: 'debugger', title: 'Debugger', kind: 'neuron', Body: DebuggerPanel },
  { id: 'benchmark', title: 'Benchmark', kind: 'experiment', Body: BenchmarkPanel },
  { id: 'benchmarksuite', title: 'Benchmark Suite', kind: 'experiment', Body: BenchmarkSuitePanel },
  { id: 'models', title: 'Models', kind: 'model', Body: ModelsCatalogPanel },
  { id: 'gpt2explorer', title: 'GPT-2 Neuron Explorer', kind: 'neuron', Body: Gpt2NeuronExplorerPanel },

  /* ---- Existing tools ---- */
  { id: 'aiassistant', title: 'AI Research Assistant', kind: 'session', Body: AIResearchAssistantView },
  { id: 'experiments', title: 'Experiments', kind: 'experiment', Body: ExperimentsView },
  { id: 'reasoning', title: 'Reasoning', kind: 'session', Body: ReasoningTraceView },
  { id: 'evidencefusion', title: 'Evidence Fusion', kind: 'session', Body: EvidenceFusionView },
  { id: 'analytics', title: 'Analytics', kind: 'session', Body: ResearchAnalyticsView },
  { id: 'health', title: 'Health', kind: 'session', Body: ScientificHealthView },
  { id: 'campaigns', title: 'Campaigns', kind: 'session', Body: CampaignWorkspaceView },
  { id: 'workspace', title: 'Workspace', kind: 'workspace', Body: Workspace },
  { id: 'prompts', title: 'Prompts', kind: 'prompt', Body: PromptsView },
  { id: 'build', title: 'Build', kind: 'workspace', Body: BuildLog },
  { id: 'sessions', title: 'Sessions', kind: 'session', Body: SessionsView },
  { id: 'reports', title: 'Reports', kind: 'paper', Body: ReportsView },
  { id: 'plugins', title: 'Plugins', kind: 'workspace', Body: PluginSDKView },
  { id: 'notebook', title: 'Research Notebook', kind: 'note', Body: ResearchNotebook },
  { id: 'labnotebook', title: 'Lab Notebook', kind: 'note', Body: ExperimentNotebook },
  { id: 'reproduction', title: 'Paper Reproduction', kind: 'paper', Body: PaperReproductionView },
  { id: 'settings', title: 'Settings', kind: 'workspace', Body: Settings },
  { id: 'logging', title: 'Logging', kind: 'session', Body: Logging },
  { id: 'projects', title: 'Projects', kind: 'workspace', Body: Projects },
  { id: 'recent', title: 'Recent Files', kind: 'session', Body: RecentFiles },
];

for (const tool of TOOLS) {
  pluginRegistry.register({
    id: tool.id,
    title: tool.title,
    icon: 'Tool',
    category: 'tool',
    resourceKinds: [tool.kind],
    defaultDock: 'center',
    fullWidth: true,
    Body: tool.Body,
  });
}