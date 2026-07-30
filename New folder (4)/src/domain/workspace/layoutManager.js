import { eventBus } from '../../utils/eventBus';

/**
 * LayoutManager - Saves, restores, and versions multi-panel workspace layouts.
 */
class LayoutManager {
  constructor() {
    this.layout = {
      version: '2.0.0',
      activePanels: ['inference-timeline', 'sae-feature', 'circuit-graph', 'logit-lens'],
      dockPosition: 'bottom',
      panelOrder: [10, 20, 30, 50],
    };
  }

  getLayout() {
    return { ...this.layout };
  }

  saveLayout(newPanels) {
    this.layout.activePanels = newPanels;
    this.layout.timestamp = new Date().toISOString();
    eventBus.emit('layout:saved', this.getLayout());
    return this.getLayout();
  }

  restoreLayout(savedLayout) {
    if (savedLayout && savedLayout.activePanels) {
      this.layout = { ...savedLayout };
      eventBus.emit('layout:restored', this.getLayout());
    }
    return this.getLayout();
  }
}

export const layoutManager = new LayoutManager();
