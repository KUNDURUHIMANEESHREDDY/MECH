'use strict';

/**
 * EventBus - Simple, fast Pub/Sub event dispatcher for decouple UI components from polling.
 */
class EventBus {
  constructor() {
    this.listeners = new Map();
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  emit(event, payload) {
    if (this.listeners.has(event)) {
      for (const callback of this.listeners.get(event)) {
        try {
          callback(payload);
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error(`[EventBus] Error in listener for ${event}:`, err);
        }
      }
    }
  }

  clear() {
    this.listeners.clear();
  }
}

export const eventBus = new EventBus();
