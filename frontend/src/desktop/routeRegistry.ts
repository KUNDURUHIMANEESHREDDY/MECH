import type { Component } from 'vue';

export type RouteRequirement = 'local' | 'backend' | 'model';
export type RouteGroup = 'Explore' | 'Develop' | 'Research' | 'General';
export type VueLoader = () => Promise<{ default: Component }>;

export interface DesktopRoute {
  id: string;
  label: string;
  group: RouteGroup;
  requirement: RouteRequirement;
  legacy: boolean;
  load: VueLoader;
  aliases?: string[];
}

export const ROUTES: DesktopRoute[] = [
  {
    id: 'explorer',
    label: 'Model Explorer',
    group: 'Explore',
    requirement: 'model',
    legacy: true,
    load: () => import('../components/ModelExplorerView.vue'),
  },
  {
    id: 'gpt2',
    label: 'GPT-2 Live',
    group: 'Explore',
    requirement: 'model',
    legacy: true,
    load: () => import('../components/Gpt2View.vue'),
  },
  {
    id: 'gpt2explorer',
    label: 'GPT-2 Neuron Explorer',
    group: 'Explore',
    requirement: 'model',
    legacy: true,
    aliases: ['gpt2-neuron-explorer', 'gpt2-neuron'],
    load: () => import('../components/ModelExplorerView.vue'),
  },
  {
    id: 'transformer',
    label: 'Transformer Visualizer',
    group: 'Explore',
    requirement: 'model',
    legacy: true,
    load: () => import('../components/TransformerVisualizer.vue'),
  },
  {
    id: 'transformerExplorer',
    label: 'Transformer Explorer',
    group: 'Explore',
    requirement: 'model',
    legacy: true,
    aliases: ['transformer-explorer', 'transformer_explorer'],
    load: () => import('../components/visualizations/TransformerExplorer.vue'),
  },
  {
    id: 'workspace',
    label: 'Workspace',
    group: 'Explore',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/CampaignWorkspaceView.vue'),
  },
  {
    id: 'models',
    label: 'Models',
    group: 'Develop',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/ModelsView.vue'),
  },
  {
    id: 'prompts',
    label: 'Prompts',
    group: 'Develop',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/PromptsView.vue'),
  },
  {
    id: 'debugger',
    label: 'Debugger',
    group: 'Develop',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/DebuggerView.vue'),
  },
  {
    id: 'experiments',
    label: 'Experiments',
    group: 'Research',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/ExperimentsView.vue'),
  },
  {
    id: 'sessions',
    label: 'Sessions',
    group: 'Research',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/SessionsView.vue'),
  },
  {
    id: 'reports',
    label: 'Reports',
    group: 'Research',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/ReportsView.vue'),
  },
  {
    id: 'settings',
    label: 'Settings',
    group: 'General',
    requirement: 'local',
    legacy: true,
    load: () => import('../components/Settings.vue'),
  },
  {
    id: 'logging',
    label: 'Logging',
    group: 'General',
    requirement: 'local',
    legacy: true,
    load: () => import('../components/Logging.vue'),
  },
  {
    id: 'build',
    label: 'Build',
    group: 'Develop',
    requirement: 'local',
    legacy: true,
    load: () => import('../components/BuildLog.vue'),
  },
  {
    id: 'neuralexplorer',
    label: 'Neural Explorer',
    group: 'Explore',
    requirement: 'model',
    legacy: true,
    aliases: ['neural-explorer'],
    load: () => import('../components/NeuralExplorerView.vue'),
  },
  {
    id: 'benchmark',
    label: 'Benchmark',
    group: 'Develop',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/BenchmarkDashboard.vue'),
  },
  {
    id: 'benchmarksuite',
    label: 'Benchmark Suite',
    group: 'Develop',
    requirement: 'backend',
    legacy: true,
    aliases: ['benchmark-suite'],
    load: () => import('../components/BenchmarkSuiteView.vue'),
  },
  {
    id: 'knowledgegraph',
    label: 'Knowledge Graph',
    group: 'Explore',
    requirement: 'backend',
    legacy: true,
    aliases: ['knowledge-graph'],
    load: () => import('../components/KnowledgeGraphView.vue'),
  },
  {
    id: 'circuitexplorer',
    label: 'Circuit Explorer',
    group: 'Explore',
    requirement: 'model',
    legacy: true,
    aliases: ['circuit-explorer'],
    load: () => import('../components/CircuitExplorerView.vue'),
  },
  {
    id: 'reasoning',
    label: 'Reasoning',
    group: 'Research',
    requirement: 'model',
    legacy: true,
    load: () => import('../components/ReasoningTraceView.vue'),
  },
  {
    id: 'evidencefusion',
    label: 'Evidence Fusion',
    group: 'Research',
    requirement: 'backend',
    legacy: true,
    aliases: ['evidence-fusion'],
    load: () => import('../components/EvidenceFusionView.vue'),
  },
  {
    id: 'campaigns',
    label: 'Campaigns',
    group: 'General',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/CampaignWorkspaceView.vue'),
  },
  {
    id: 'analytics',
    label: 'Analytics',
    group: 'Research',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/ResearchAnalyticsView.vue'),
  },
  {
    id: 'health',
    label: 'Health',
    group: 'Research',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/ScientificHealthView.vue'),
  },
  {
    id: 'plugins',
    label: 'Plugins',
    group: 'General',
    requirement: 'local',
    legacy: true,
    load: () => import('../components/PluginSDKView.vue'),
  },
  {
    id: 'notebook',
    label: 'Research Notebook',
    group: 'General',
    requirement: 'local',
    legacy: true,
    load: () => import('../components/ResearchNotebook.vue'),
  },
  {
    id: 'labnotebook',
    label: 'Lab Notebook',
    group: 'General',
    requirement: 'local',
    legacy: true,
    aliases: ['lab-notebook'],
    load: () => import('../components/ExperimentNotebook.vue'),
  },
  {
    id: 'reproduction',
    label: 'Paper Reproduction',
    group: 'General',
    requirement: 'backend',
    legacy: true,
    load: () => import('../components/PaperReproductionView.vue'),
  },
  {
    id: 'projects',
    label: 'Projects',
    group: 'General',
    requirement: 'local',
    legacy: true,
    load: () => import('../components/Projects.vue'),
  },
  {
    id: 'recent',
    label: 'Recent Files',
    group: 'General',
    requirement: 'local',
    legacy: true,
    aliases: ['recent-files'],
    load: () => import('../components/RecentFiles.vue'),
  },
  {
    id: 'society',
    label: 'Society',
    group: 'Research',
    requirement: 'backend',
    legacy: false,
    load: () => import('../components/ResearchSocietyView.vue'),
  },
];

export const LEGACY_ROUTE_IDS = ROUTES.filter((route) => route.legacy).map((route) => route.id);

const routeById = new Map(ROUTES.map((route) => [route.id, route]));
const routeByAlias = new Map(
  ROUTES.flatMap((route) => (route.aliases ?? []).map((alias) => [alias, route] as const)),
);

export function getRouteById(id: string): DesktopRoute | null {
  return routeById.get(id) ?? routeByAlias.get(id) ?? null;
}

export function isLegacyRoute(id: string): boolean {
  return getRouteById(id)?.legacy === true;
}
