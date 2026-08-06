<template>
  <div class="flex h-screen bg-[var(--bg)] text-[var(--ink)] font-sans overflow-hidden app"
    :class="{'activity-collapsed': activityCollapsed}">
    <ActivityBar
      :active="activePage"
      :collapsed="activityCollapsed"
      @select="setActivePage"
      @toggle="activityCollapsed = !activityCollapsed"
      @toggle-sidebar="sidebarCollapsed = !sidebarCollapsed"
    />
    <Sidebar
      :active="activePage"
      :collapsed="sidebarCollapsed"
      @select="setActivePage"
      @toggle="sidebarCollapsed = !sidebarCollapsed"
    />
    <div class="flex-1 flex flex-col min-w-0">
      <Topbar
        :crumb="pageLabels[activePage]?.label ?? 'MECH'"
        :python-status="pythonStatus"
        :collapsed="activityCollapsed"
        @toggle-cmd-palette="showCmdPalette = !showCmdPalette"
        @toggle-activity="activityCollapsed = !activityCollapsed"
      />
      <div class="flex-1 overflow-auto p-4">
        <template v-if="modelLoaded">
          <component
            :is="currentPageComponent"
            :key="activePage"
            v-bind="currentPageProps"
          />
        </template>
        <div v-else class="flex flex-col items-center justify-center h-full gap-4">
          <div class="text-[var(--ink-muted)] text-sm">Choose a model to load</div>
          <div class="flex gap-2 flex-wrap justify-center">
            <button
              v-for="name in availableModels"
              :key="name"
              @click="loadModel(name)"
              class="px-4 py-2 rounded-lg border-none bg-transparent text-[var(--text)] text-sm font-medium hover:bg-[var(--bg-hover)] transition-colors"
            >
              Load {{ name }}
            </button>
          </div>
        </div>
      </div>
    </div>
    <StatusBar :model-name="modelName" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, shallowRef } from 'vue';
import { useAppStore } from './store/app';
import { api } from './services/api';
import Sidebar from './components/Sidebar.vue';
import Topbar from './components/Topbar.vue';
import ActivityBar from './components/ActivityBar.vue';
import StatusBar from './components/StatusBar.vue';

const store = useAppStore();

const activePage = computed({
  get: () => store.activePage,
  set: (v: string) => store.setActivePage(v),
});
const sidebarCollapsed = computed({
  get: () => store.sidebarCollapsed,
  set: (v: boolean) => store.set({ sidebarCollapsed: v }),
});
const activityCollapsed = computed({
  get: () => store.activityCollapsed,
  set: (v: boolean) => store.set({ activityCollapsed: v }),
});

const showCmdPalette = ref(false);
const modelLoaded = ref(false);
const modelName = ref('gpt2');
const availableModels = ref<string[]>([]);
const pythonStatus = ref<'connected' | 'offline' | 'connecting'>('connecting');

const PAGE_COMPONENTS: Record<string, () => Promise<{ default: any }>> = {};

const currentPageComponent = shallowRef<any>(null);
const currentPageProps = ref<Record<string, unknown>>({});

const pageLabels: Record<string, { label: string }> = {
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

function setActivePage(page: string) {
  store.setActivePage(page);
  if (typeof window !== 'undefined' && window.location.hash.slice(1) !== page) {
    window.location.hash = page;
  }
}

async function loadModel(name: string) {
  try {
    pythonStatus.value = 'connecting';
    await api.loadModel(name);
    modelLoaded.value = true;
    modelName.value = name;
    pythonStatus.value = 'connected';
  } catch {
    pythonStatus.value = 'offline';
  }
}

async function loadPageComponent(page: string) {
  const map: Record<string, () => Promise<{ default: any }>> = {
    explorer: () => import('./components/ModelExplorerView.vue'),
    gpt2: () => import('./components/Gpt2View.vue'),
    neuralexplorer: () => import('./components/NeuralExplorerView.vue'),
    transformerExplorer: () => import('./components/visualizations/TransformerExplorer.vue'),
    settings: () => import('./components/Settings.vue'),
    experiments: () => import('./components/ExperimentsView.vue'),
    sessions: () => import('./components/SessionsView.vue'),
    reports: () => import('./components/ReportsView.vue'),
    models: () => import('./components/ModelsView.vue'),
    prompts: () => import('./components/PromptsView.vue'),
    debugger: () => import('./components/DebuggerView.vue'),
    benchmark: () => import('./components/BenchmarkDashboard.vue'),
    benchmarksuite: () => import('./components/BenchmarkSuiteView.vue'),
    knowledgegraph: () => import('./components/KnowledgeGraphView.vue'),
    circuitexplorer: () => import('./components/CircuitExplorerView.vue'),
    reasoning: () => import('./components/ReasoningTraceView.vue'),
    evidencefusion: () => import('./components/EvidenceFusionView.vue'),
    campaigns: () => import('./components/CampaignWorkspaceView.vue'),
    analytics: () => import('./components/ResearchAnalyticsView.vue'),
    health: () => import('./components/ScientificHealthView.vue'),
    plugins: () => import('./components/PluginSDKView.vue'),
    notebook: () => import('./components/ResearchNotebook.vue'),
    labnotebook: () => import('./components/ExperimentNotebook.vue'),
    reproduction: () => import('./components/PaperReproductionView.vue'),
    projects: () => import('./components/Projects.vue'),
    recent: () => import('./components/RecentFiles.vue'),
    logging: () => import('./components/Logging.vue'),
    build: () => import('./components/BuildLog.vue'),
    workspace: () => import('./components/CampaignWorkspaceView.vue'),
     transformer: () => import('./components/TransformerVisualizer.vue'),
     gpt2explorer: () => import('./components/Gpt2View.vue'),
   };
  const loader = map[page];
  if (loader) {
    try {
      const mod = await loader();
      currentPageComponent.value = mod.default;
    } catch {
      currentPageComponent.value = null;
    }
  } else {
    currentPageComponent.value = null;
  }
}

watch(activePage, (p) => loadPageComponent(p), { immediate: true });

onMounted(async () => {
  const hash = window.location.hash.slice(1);
  if (hash && pageLabels[hash]) {
    store.setActivePage(hash);
  }
  window.addEventListener('hashchange', () => {
    const next = window.location.hash.slice(1);
    if (next && pageLabels[next]) store.setActivePage(next);
  });

  try {
    const res = await api.listModels();
    availableModels.value = res.models ?? [];
    pythonStatus.value = 'connected';
  } catch {
    pythonStatus.value = 'offline';
  }
});
</script>
