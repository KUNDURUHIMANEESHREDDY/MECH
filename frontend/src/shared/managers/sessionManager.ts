import { persistence } from '../api/persistence';
import { WorkspaceState } from '../types';

export interface SessionSnapshot {
  workspaceId: string;
  activeModel: string | null;
  activeSessionId: string | null;
  windows: any[];
  camera: { x: number; y: number; zoom: number };
  notes: any[];
  timeline: any[];
  console: any[];
  visiblePanels: Record<string, boolean>;
  savedAt: number;
}

class SessionManager {
  private currentWorkspace = 'default';

  async saveSession(workspaceState: WorkspaceState): Promise<void> {
    const snapshot: SessionSnapshot = {
      workspaceId: workspaceState.activeWorkspace || this.currentWorkspace,
      activeModel: workspaceState.activeModel,
      activeSessionId: workspaceState.activeSessionId,
      windows: workspaceState.windows || [],
      camera: workspaceState.camera || { x: 0, y: 0, zoom: 1 },
      notes: workspaceState.notes || [],
      timeline: workspaceState.timeline || [],
      console: workspaceState.console || [],
      visiblePanels: workspaceState.visiblePanels || {},
      savedAt: Date.now(),
    };

    const key = `session:${snapshot.workspaceId}`;
    await persistence.set(key, snapshot);
  }

  async loadSession(workspaceId: string = 'default'): Promise<SessionSnapshot | null> {
    this.currentWorkspace = workspaceId;
    const key = `session:${workspaceId}`;
    return await persistence.get<SessionSnapshot | null>(key, null);
  }
}

export const sessionManager = new SessionManager();
