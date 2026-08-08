import { useWorkspaceStore } from '../stores/workspace';

export class WindowManager {
  detachPanel(panelId: string, pos?: { x: number; y: number }) {
    useWorkspaceStore.getState().detachPanelToWindow(panelId, pos);
  }

  closeWindow(windowId: string) {
    useWorkspaceStore.getState().closeWindow(windowId);
  }

  bringToFront(windowId: string) {
    const windows = useWorkspaceStore.getState().windows;
    const maxZ = Math.max(0, ...windows.map((w) => w.zIndex));
    useWorkspaceStore.getState().updateWindow(windowId, { zIndex: maxZ + 1 });
  }

  updateBounds(windowId: string, bounds: { x?: number; y?: number; width?: number; height?: number }) {
    useWorkspaceStore.getState().updateWindow(windowId, bounds);
  }

  getWindows() {
    return useWorkspaceStore.getState().windows;
  }
}

export const windowManager = new WindowManager();
