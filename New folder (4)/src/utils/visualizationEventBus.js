/**
 * Visualization Event Bus.
 * Broadcasts interaction events: SelectionChanged, CameraMoved, PlaybackStarted, PlaybackPaused, GraphUpdated, ExportFinished.
 */

class VisualizationEventBus {
  constructor() {
    this.listeners = {};
  }

  on(event, handler) {
    if (!this.listeners[event]) this.listeners[event] = new Set();
    this.listeners[event].add(handler);
    return () => this.listeners[event]?.delete(handler);
  }

  emit(event, payload) {
    if (this.listeners[event]) {
      this.listeners[event].forEach((handler) => handler(payload));
    }
  }
}

export const visualizationEventBus = new VisualizationEventBus();
