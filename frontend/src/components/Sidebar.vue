<template>
  <div
    class="flex flex-col bg-[var(--bg-sidebar)] transition-all duration-200 overflow-hidden shrink-0"
    :class="collapsed ? 'w-0 border-r-0' : 'w-[264px] border-r border-[var(--border)]'"
  >
    <div v-if="!collapsed" class="flex items-center justify-between px-4 pt-3 pb-2 border-b border-[var(--border)]">
      <h1 class="text-[var(--ink)] text-lg font-bold tracking-tight m-0">Explorer</h1>
      <button
        title="Collapse sidebar"
        @click="$emit('toggle')"
        class="w-8 h-8 rounded-full border-none bg-transparent text-[var(--ink-muted-80)] cursor-pointer flex items-center justify-center hover:text-[var(--ink)]"
      >
        <ChevronLeft :size="16" />
      </button>
    </div>

    <div v-if="!collapsed" class="flex-1 overflow-y-auto px-4 py-2">
      <div class="flex items-center gap-2 bg-[var(--canvas)] rounded-full px-4 py-1.5 mb-4 border border-[var(--border)]">
        <Search :size="13" class="text-[var(--ink-muted-48)]" />
        <input
          v-model="query"
          type="text"
          placeholder="Search views..."
          class="border-none outline-none bg-transparent text-[var(--ink)] text-sm w-full py-1 font-sans"
        />
      </div>

      <div v-for="section in sections" :key="section.title" class="mb-4">
        <div class="text-[var(--ink-muted-48)] text-[10px] font-semibold tracking-wider uppercase py-1">
          {{ section.title }}
        </div>
        <div class="flex flex-col gap-0.5">
          <div
             v-for="key in section.items.filter(matches)"
             :key="key"
             :data-testid="`nav-${key}`"
             role="button"
             tabindex="0"
             class="flex items-center gap-2 px-2 py-1 rounded cursor-pointer transition-colors duration-150 text-sm"
             :class="active === key
               ? 'bg-[var(--accent-soft)] text-[var(--primary)] font-semibold'
               : 'text-[var(--ink)] hover:bg-[var(--accent-soft)]'"
             @click="$emit('select', key)"
             @keydown.enter="$emit('select', key)"
             @keydown.space.prevent="$emit('select', key)"
           >
            <component :is="icons[key]" v-if="icons[key]" :size="15" :stroke-width="1.75" />
            <span>{{ getLabel(key) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, markRaw } from 'vue';
import {
  Network, Workflow, BrainCircuit, Share2, Waypoints,
  Boxes, Terminal, Bug, Hammer, Gauge, ListChecks,
  FlaskConical, History, FileText, GitBranch, Scale,
  TrendingUp, HeartPulse, Rocket, Puzzle, BookOpen,
  ClipboardList, CopyCheck, Settings, ScrollText,
  FolderKanban, Clock, ChevronLeft, ChevronRight, Search,
  Activity,
} from 'lucide-vue-next';

defineProps<{
  active: string;
  collapsed: boolean;
}>();

defineEmits<{
  select: [page: string];
  toggle: [];
}>();

const query = ref('');

const sections = [
  { title: 'Explore', items: ['explorer', 'gpt2', 'transformer', 'workspace', 'neuralexplorer', 'knowledgegraph', 'circuitexplorer'] },
  { title: 'Develop', items: ['models', 'prompts', 'debugger', 'build', 'benchmark', 'benchmarksuite'] },
  { title: 'Research', items: ['experiments', 'sessions', 'reports', 'reasoning', 'evidencefusion', 'analytics', 'health'] },
  { title: 'General', items: ['campaigns', 'plugins', 'notebook', 'labnotebook', 'reproduction', 'settings', 'logging', 'projects', 'recent'] },
];

const labels: Record<string, string> = {
  explorer: 'Model Explorer', gpt2: 'GPT-2 Live', transformer: 'Transformer Visualizer',
  workspace: 'Workspace', neuralexplorer: 'Neural Explorer', knowledgegraph: 'Knowledge Graph',
  circuitexplorer: 'Circuit Explorer', models: 'Models', prompts: 'Prompts', debugger: 'Debugger',
  build: 'Build Log', benchmark: 'Benchmark', benchmarksuite: 'Benchmark Suite',
  experiments: 'Experiments', sessions: 'Sessions', reports: 'Reports',
  reasoning: 'Reasoning Trace', evidencefusion: 'Evidence Fusion', analytics: 'Analytics',
  health: 'Health', campaigns: 'Campaigns', plugins: 'Plugins', notebook: 'Research Notebook',
  labnotebook: 'Lab Notebook', reproduction: 'Paper Reproduction', settings: 'Settings',
  logging: 'Logging', projects: 'Projects', recent: 'Recent Files',
};

const icons: Record<string, any> = {
  explorer: markRaw(Network), gpt2: markRaw(Activity), transformer: markRaw(Workflow),
  workspace: markRaw(BrainCircuit), neuralexplorer: markRaw(BrainCircuit),
  knowledgegraph: markRaw(Share2), circuitexplorer: markRaw(Waypoints),
  models: markRaw(Boxes), prompts: markRaw(Terminal), debugger: markRaw(Bug),
  build: markRaw(Hammer), benchmark: markRaw(Gauge), benchmarksuite: markRaw(ListChecks),
  experiments: markRaw(FlaskConical), sessions: markRaw(History), reports: markRaw(FileText),
  reasoning: markRaw(GitBranch), evidencefusion: markRaw(Scale), analytics: markRaw(TrendingUp),
  health: markRaw(HeartPulse), campaigns: markRaw(Rocket), plugins: markRaw(Puzzle),
  notebook: markRaw(BookOpen), labnotebook: markRaw(ClipboardList),
  reproduction: markRaw(CopyCheck), settings: markRaw(Settings), logging: markRaw(ScrollText),
  projects: markRaw(FolderKanban), recent: markRaw(Clock),
};

function getLabel(key: string): string {
  return labels[key] || key;
}

function matches(key: string): boolean {
  const q = query.value.trim().toLowerCase();
  return q === '' || getLabel(key).toLowerCase().includes(q);
}
</script>
