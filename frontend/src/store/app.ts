import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useAppStore = defineStore('app', () => {
  const activePage = ref('explorer');
  const sidebarCollapsed = ref(false);
  const activityCollapsed = ref(false);

  function setActivePage(page: string) {
    activePage.value = page;
  }

  function set(partial: Partial<{ activePage: string; sidebarCollapsed: boolean; activityCollapsed: boolean }>) {
    if (partial.activePage !== undefined) activePage.value = partial.activePage;
    if (partial.sidebarCollapsed !== undefined) sidebarCollapsed.value = partial.sidebarCollapsed;
    if (partial.activityCollapsed !== undefined) activityCollapsed.value = partial.activityCollapsed;
  }

  return { activePage, sidebarCollapsed, activityCollapsed, setActivePage, set };
});
