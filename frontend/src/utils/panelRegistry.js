'use strict';

import { Pin } from 'lucide-react';
import { eventBus } from './eventBus';

/**
 * PanelRegistry - Central registry for visualization and debugger panels.
 * Allows core and plugin panels to auto-register with DockManager.
 */
class PanelRegistry {
  constructor() {
    this.panels = new Map();
  }

  register(panel) {
    if (!panel || !panel.id || !panel.name) {
      throw new Error('Panel registration requires id and name');
    }
    this.panels.set(panel.id, {
      id: panel.id,
      name: panel.name,
      icon: panel.icon || Pin,
      desc: panel.desc || '',
      component: panel.component || null,
      order: panel.order || 100
    });
    eventBus.emit('panels:updated', this.getPanels());
  }

  unregister(id) {
    if (this.panels.has(id)) {
      this.panels.delete(id);
      eventBus.emit('panels:updated', this.getPanels());
    }
  }

  getPanels() {
    return Array.from(this.panels.values()).sort((a, b) => a.order - b.order);
  }

  getPanel(id) {
    return this.panels.get(id) || null;
  }
}

export const panelRegistry = new PanelRegistry();
