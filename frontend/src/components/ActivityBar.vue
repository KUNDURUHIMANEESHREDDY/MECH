<template>
  <div
    class="flex flex-col items-center py-2 border-r border-[var(--hairline)] bg-[var(--bg-sidebar)] transition-all duration-200 overflow-hidden"
    :class="collapsed ? 'w-0' : 'w-[52px]'"
  >
    <div class="flex flex-col items-center gap-2 flex-1">
      <div
        class="w-8 h-8 rounded-full bg-[var(--primary)] text-white flex items-center justify-center text-sm font-bold mb-3"
      >
        M
      </div>
      <button
        v-for="item in topIcons"
        :key="item.id"
        :title="item.label"
        @click="item.section && $emit('select', item.section)"
        class="w-9 h-9 rounded-full border-none flex items-center justify-center cursor-pointer transition-colors duration-150"
        :class="active === item.section
          ? 'bg-[var(--primary)] text-white'
          : 'bg-transparent text-[var(--ink-muted-48)] hover:text-[var(--ink-muted-80)]'"
      >
        <component :is="item.icon" :size="18" :stroke-width="1.75" />
      </button>
    </div>
    <button
      title="Toggle sidebar"
      @click="$emit('toggle-sidebar')"
      class="w-9 h-9 rounded-full border-none bg-transparent text-[var(--ink-muted-48)] cursor-pointer flex items-center justify-center hover:text-[var(--ink-muted-80)]"
    >
      <SidebarIcon :size="18" :stroke-width="1.75" />
    </button>
  </div>
</template>

<script setup lang="ts">
import { markRaw } from 'vue';
import { LayoutGrid, Search, Bug, FileCode2, Sidebar as SidebarIcon } from 'lucide-vue-next';

defineProps<{
  active: string;
  collapsed: boolean;
}>();

defineEmits<{
  select: [page: string];
  toggle: [];
  'toggle-sidebar': [];
}>();

const topIcons = [
  { id: 'explorer', icon: markRaw(LayoutGrid), label: 'Explorer', section: 'explorer' },
  { id: 'search', icon: markRaw(Search), label: 'Search', section: '' },
  { id: 'debug', icon: markRaw(Bug), label: 'Debug', section: 'debugger' },
  { id: 'code', icon: markRaw(FileCode2), label: 'Code', section: '' },
];
</script>
