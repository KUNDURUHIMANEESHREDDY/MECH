import { computed, reactive, ref } from 'vue';
import { defineStore } from 'pinia';
import { getRouteById } from '../desktop/routeRegistry';

export const MIN_WINDOW_WIDTH = 320;
export const MIN_WINDOW_HEIGHT = 220;

export interface WindowGeometry {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DesktopBounds {
  width: number;
  height: number;
}

export interface DesktopWindowState extends WindowGeometry {
  id: string;
  routeId: string;
  closed: boolean;
  minimized: boolean;
  maximized: boolean;
  zIndex: number;
}

export const DEFAULT_DESKTOP_BOUNDS: DesktopBounds = { width: 1200, height: 800 };
export const DEFAULT_WINDOW_GEOMETRY: WindowGeometry = { x: 48, y: 40, width: 860, height: 620 };

function finiteOr(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function normalizeBounds(bounds?: DesktopBounds): DesktopBounds {
  return {
    width: Math.max(1, finiteOr(bounds?.width, DEFAULT_DESKTOP_BOUNDS.width)),
    height: Math.max(1, finiteOr(bounds?.height, DEFAULT_DESKTOP_BOUNDS.height)),
  };
}

function clampGeometry(geometry: Partial<WindowGeometry>, bounds: DesktopBounds): WindowGeometry {
  const width = Math.min(
    Math.max(MIN_WINDOW_WIDTH, finiteOr(geometry.width, MIN_WINDOW_WIDTH)),
    Math.max(MIN_WINDOW_WIDTH, bounds.width),
  );
  const height = Math.min(
    Math.max(MIN_WINDOW_HEIGHT, finiteOr(geometry.height, MIN_WINDOW_HEIGHT)),
    Math.max(MIN_WINDOW_HEIGHT, bounds.height),
  );

  const x = finiteOr(geometry.x, 0);
  const y = finiteOr(geometry.y, 0);
  const maxX = Math.max(0, bounds.width - width);
  const maxY = Math.max(0, bounds.height - height);

  return {
    width,
    height,
    x: x >= 0 && x <= maxX ? x : 0,
    y: y >= 0 && y <= maxY ? y : 0,
  };
}

export const useDesktopStore = defineStore('desktop', () => {
  const windows = reactive<Record<string, DesktopWindowState>>({});
  const activeWindowId = ref<string | null>(null);
  let nextZIndex = 1;

  const openWindows = computed(() => Object.values(windows).filter((window) => !window.closed));
  const focusedWindow = computed(() => {
    const id = activeWindowId.value;
    return id ? windows[id] ?? null : null;
  });

  function getWindow(routeId: string): DesktopWindowState | null {
    const route = getRouteById(routeId);
    return route ? windows[route.id] ?? null : null;
  }

  function ensureWindow(routeId: string): DesktopWindowState | null {
    const route = getRouteById(routeId);
    if (!route) return null;

    const existing = windows[route.id];
    if (existing) return existing;

    const windowState: DesktopWindowState = {
      id: route.id,
      routeId: route.id,
      ...clampGeometry(DEFAULT_WINDOW_GEOMETRY, DEFAULT_DESKTOP_BOUNDS),
      closed: true,
      minimized: false,
      maximized: false,
      zIndex: nextZIndex++,
    };
    windows[route.id] = windowState;
    return windowState;
  }

  function focusState(windowState: DesktopWindowState): void {
    windowState.zIndex = nextZIndex++;
    activeWindowId.value = windowState.routeId;
  }

  function visibleFallback(): string | null {
    const candidate = Object.values(windows)
      .filter((windowState) => !windowState.closed && !windowState.minimized)
      .sort((left, right) => right.zIndex - left.zIndex)[0];
    return candidate?.routeId ?? null;
  }

  function openWindow(routeId: string): boolean {
    const windowState = ensureWindow(routeId);
    if (!windowState) return false;

    windowState.closed = false;
    windowState.minimized = false;
    focusState(windowState);
    return true;
  }

  function closeWindow(routeId: string): boolean {
    const windowState = getWindow(routeId);
    if (!windowState) return false;

    windowState.closed = true;
    windowState.minimized = false;
    if (activeWindowId.value === windowState.routeId) {
      activeWindowId.value = visibleFallback();
    }
    return true;
  }

  function focusWindow(routeId: string): boolean {
    const windowState = getWindow(routeId);
    if (!windowState || windowState.closed) return false;

    windowState.minimized = false;
    focusState(windowState);
    return true;
  }

  function minimizeWindow(routeId: string): boolean {
    const windowState = getWindow(routeId);
    if (!windowState || windowState.closed) return false;

    windowState.minimized = true;
    if (activeWindowId.value === windowState.routeId) {
      activeWindowId.value = visibleFallback();
    }
    return true;
  }

  function toggleMaximize(routeId: string, bounds?: DesktopBounds): boolean {
    const windowState = getWindow(routeId);
    if (!windowState || windowState.closed) return false;

    if (bounds) Object.assign(windowState, clampGeometry(windowState, normalizeBounds(bounds)));
    windowState.maximized = !windowState.maximized;
    windowState.minimized = false;
    focusState(windowState);
    return windowState.maximized;
  }

  function setGeometry(
    routeId: string,
    geometry: Partial<WindowGeometry>,
    bounds?: DesktopBounds,
  ): boolean {
    const windowState = ensureWindow(routeId);
    if (!windowState) return false;

    Object.assign(
      windowState,
      clampGeometry({ ...windowState, ...geometry }, normalizeBounds(bounds)),
    );
    windowState.maximized = false;
    return true;
  }

  function tileWindows(bounds?: DesktopBounds, gap = 12): void {
    const available = openWindows.value;
    if (available.length === 0) return;

    const desktop = normalizeBounds(bounds);
    const spacing = Math.max(0, finiteOr(gap, 12));
    const columns = Math.min(
      Math.max(1, Math.floor((desktop.width + spacing) / (MIN_WINDOW_WIDTH + spacing))),
      Math.ceil(Math.sqrt(available.length)),
    );
    const rows = Math.ceil(available.length / columns);
    const cellWidth = (desktop.width - spacing * (columns + 1)) / columns;
    const cellHeight = (desktop.height - spacing * (rows + 1)) / rows;

    available.forEach((windowState, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      setGeometry(
        windowState.routeId,
        {
          x: spacing + column * (cellWidth + spacing),
          y: spacing + row * (cellHeight + spacing),
          width: cellWidth,
          height: cellHeight,
        },
        desktop,
      );
    });
  }

  function resetWindows(): void {
    Object.keys(windows).forEach((routeId) => delete windows[routeId]);
    activeWindowId.value = null;
    nextZIndex = 1;
  }

  function getOpenWindows(): DesktopWindowState[] {
    return openWindows.value;
  }

  function getFocusedWindow(): DesktopWindowState | null {
    return focusedWindow.value;
  }

  return {
    windows,
    activeWindowId,
    openWindows,
    focusedWindow,
    openWindow,
    closeWindow,
    focusWindow,
    minimizeWindow,
    toggleMaximize,
    setGeometry,
    tileWindows,
    resetWindows,
    getOpenWindows,
    getFocusedWindow,
    getWindow,
  };
});
