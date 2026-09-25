import { beforeEach, describe, expect, it } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { useDesktopStore } from '../../src/stores/desktop';

const bounds = { width: 1200, height: 760 };

describe('desktop window state', () => {
  beforeEach(() => setActivePinia(createPinia()));

  it('opens and focuses a tool window without requiring a model', () => {
    const store = useDesktopStore();
    expect(store.openWindow('settings')).toBe(true);

    expect(store.activeWindowId).toBe('settings');
    expect(store.windows.settings.closed).toBe(false);
    expect(store.windows.settings.routeId).toBe('settings');
  });

  it('opens Society as an additive desktop tool', () => {
    const store = useDesktopStore();
    expect(store.openWindow('society')).toBe(true);
    expect(store.windows.society.routeId).toBe('society');
  });

  it('closes, restores, and focuses windows predictably', () => {
    const store = useDesktopStore();
    store.openWindow('explorer');
    store.openWindow('settings');
    store.closeWindow('settings');

    expect(store.windows.settings.closed).toBe(true);
    expect(store.activeWindowId).toBe('explorer');
    expect(store.openWindow('settings')).toBe(true);
    expect(store.activeWindowId).toBe('settings');
  });

  it('tiles open windows inside the desktop bounds', () => {
    const store = useDesktopStore();
    store.openWindow('explorer');
    store.openWindow('circuitexplorer');
    store.openWindow('society');
    store.tileWindows(bounds, 14);

    expect(store.windows.explorer.x).toBe(14);
    expect(store.windows.explorer.y).toBe(14);
    expect(store.windows.circuitexplorer.x).toBeGreaterThan(store.windows.explorer.x);
    expect(store.windows.society.y).toBeGreaterThan(store.windows.explorer.y);
    expect(store.windows.society.x + store.windows.society.width).toBeLessThanOrEqual(bounds.width);
    expect(store.windows.society.y + store.windows.society.height).toBeLessThanOrEqual(bounds.height);
  });

  it('clamps geometry to desktop bounds and minimum size', () => {
    const store = useDesktopStore();
    store.openWindow('settings');
    store.setGeometry('settings', { x: -500, y: 900, width: 20, height: 10 }, bounds);

    expect(store.windows.settings.x).toBe(0);
    expect(store.windows.settings.y).toBe(0);
    expect(store.windows.settings.width).toBeGreaterThanOrEqual(320);
    expect(store.windows.settings.height).toBeGreaterThanOrEqual(220);
  });
});
