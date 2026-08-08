import { PanelPlugin } from '../shared/types';

export class PluginRegistry {
  private plugins: Map<string, PanelPlugin> = new Map();

  register(plugin: PanelPlugin): void {
    this.plugins.set(plugin.id, plugin);
  }

  get(id: string): PanelPlugin | undefined {
    return this.plugins.get(id);
  }

  list(): PanelPlugin[] {
    return Array.from(this.plugins.values());
  }

  getByResourceKind(kind: string): PanelPlugin[] {
    return this.list().filter(
      (plugin) => plugin.resourceKinds && plugin.resourceKinds.includes(kind as any)
    );
  }

  getByCategory(category: string): PanelPlugin[] {
    return this.list().filter((plugin) => plugin.category === category);
  }

  clear(): void {
    this.plugins.clear();
  }
}

export const pluginRegistry = new PluginRegistry();
