import { useWorkspaceStore } from '../stores/workspace';

export class WorkspaceManager {
  setWorkspace(id: string) {
    useWorkspaceStore.getState().setActiveWorkspace(id);
  }

  setCamera(x: number, y: number, zoom: number) {
    useWorkspaceStore.getState().updateCamera({ x, y, zoom });
  }

  getCamera() {
    return useWorkspaceStore.getState().camera;
  }

  getWorkspaceNames() {
    return useWorkspaceStore.getState().workspaceNames;
  }
}

export const workspaceManager = new WorkspaceManager();
