import { CommandEvent } from '../types';

type Handler = (event: CommandEvent) => void;

class CommandManager {
  private handlers: Map<string, Set<Handler>> = new Map();

  publish(eventOrType: CommandEvent | string, payload?: unknown) {
    const event: CommandEvent =
      typeof eventOrType === 'string'
        ? { type: eventOrType, payload: payload ?? null, timestamp: Date.now() }
        : eventOrType;

    const type = event.type;
    const handlers = this.handlers.get(type);
    if (handlers) {
      handlers.forEach((fn) => {
        try { fn(event); } catch (e) { console.error('[CommandManager] handler error', e); }
      });
    }
    // wildcard
    const all = this.handlers.get('*');
    if (all) {
      all.forEach((fn) => {
        try { fn(event); } catch (e) { console.error('[CommandManager] wildcard error', e); }
      });
    }
  }

  subscribe(type: string, handler: Handler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
    return () => this.handlers.get(type)!.delete(handler);
  }

  clear() {
    this.handlers.clear();
  }
}

export const commandManager = new CommandManager();

export const Commands = {
  neuronSelected(layer: number, head: number, neuron: number) {
    commandManager.publish({
      type: 'neuron.selected',
      payload: { layer, head, neuron },
      timestamp: Date.now()
    });
  },
  resourceOpened(resource: any) {
    commandManager.publish({
      type: 'resource.opened',
      payload: resource,
      timestamp: Date.now()
    });
  }
};
