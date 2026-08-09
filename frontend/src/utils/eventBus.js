type Handler = (event: string, data?: any) => void;

class EventBus {
  private handlers: Map<string, Set<Handler>> = new Map();

  on(event: string, handler: Handler) {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
    return () => this.handlers.get(event)!.delete(handler);
  }

  off(event: string, handler: Handler) {
    this.handlers.get(event)?.delete(handler);
  }

  emit(event: string, data?: any) {
    const handlers = this.handlers.get(event);
    if (handlers) {
      handlers.forEach((fn) => {
        try { fn(event, data); } catch (e) { console.error('[eventBus] handler error', e); }
      });
    }
  }

  clear() {
    this.handlers.clear();
  }
}

export const eventBus = new EventBus();
export const commandBus = eventBus;
