/**
 * Persistence adapter for MECH desktop application.
 * Uses Electron preload window.appApi.settings KV store if present,
 * with graceful fallback to localStorage for standalone browser dev.
 */

export const persistence = {
  async get<T = unknown>(key: string, defaultValue: T): Promise<T> {
    try {
      if (typeof window !== 'undefined' && (window as any).appApi?.settings?.get) {
        const val = await (window as any).appApi.settings.get(key);
        if (val !== undefined && val !== null) {
          return typeof val === 'string' ? JSON.parse(val) : val;
        }
      }
      const raw = localStorage.getItem(`mech:${key}`);
      if (raw) {
        return JSON.parse(raw);
      }
    } catch (err) {
      console.warn(`[persistence] Failed to get key "${key}":`, err);
    }
    return defaultValue;
  },

  async set<T = unknown>(key: string, value: T): Promise<void> {
    try {
      const jsonStr = JSON.stringify(value);
      if (typeof window !== 'undefined' && (window as any).appApi?.settings?.set) {
        await (window as any).appApi.settings.set(key, jsonStr);
      }
      localStorage.setItem(`mech:${key}`, jsonStr);
    } catch (err) {
      console.warn(`[persistence] Failed to set key "${key}":`, err);
    }
  },

  async remove(key: string): Promise<void> {
    try {
      if (typeof window !== 'undefined' && (window as any).appApi?.settings?.set) {
        await (window as any).appApi.settings.set(key, null);
      }
      localStorage.removeItem(`mech:${key}`);
    } catch (err) {
      console.warn(`[persistence] Failed to remove key "${key}":`, err);
    }
  },
};
