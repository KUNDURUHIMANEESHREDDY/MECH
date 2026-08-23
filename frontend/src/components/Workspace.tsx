import React, { useState, useCallback, useEffect } from 'react';
import { FolderTree, Activity, Library, FileText, ChevronRight, Save, RotateCcw, Clock, GripVertical, Trash2 } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useModel } from '../shared/hooks/useModel';
import { useWorkspaceStore } from '../shared/stores/workspace';
import { sessionManager } from '../shared/managers/sessionManager';

const card: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: 12,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
};
const btn: React.CSSProperties = {
  background: colors.primary,
  color: colors.onPrimary,
  border: 'none',
  borderRadius: 6,
  padding: '6px 14px',
  fontSize: 12,
  fontWeight: 600,
  cursor: 'pointer',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
};
const stat: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: '14px 16px',
  display: 'flex',
  flexDirection: 'column',
  gap: 4,
  flex: 1,
  minWidth: 140,
};

interface PanelConfig {
  id: string;
  title: string;
  visible: boolean;
  order: number;
}

const DEFAULT_PANELS: PanelConfig[] = [
  { id: 'stats', title: 'Statistics', visible: true, order: 0 },
  { id: 'workspaces', title: 'Workspaces', visible: true, order: 1 },
  { id: 'activity', title: 'Recent Activity', visible: true, order: 2 },
  { id: 'sessions', title: 'Session History', visible: true, order: 3 },
];

export const Workspace: React.FC = () => {
  const { state: model } = useModel();
  const { activeWorkspace, workspaceNames, setActiveWorkspace, timeline, notes: workspaceNotes, console: consoleLogs, addTimelineEvent } = useWorkspaceStore();
  
  const [panels, setPanels] = useState<PanelConfig[]>(() => {
    const saved = localStorage.getItem('workspace-panels');
    return saved ? JSON.parse(saved) : DEFAULT_PANELS;
  });
  const [draggedPanel, setDraggedPanel] = useState<string | null>(null);
  const [sessionHistory, setSessionHistory] = useState<Array<{ id: string; name: string; timestamp: number }>>([]);
  const [showSessionHistory, setShowSessionHistory] = useState(false);

  useEffect(() => {
    localStorage.setItem('workspace-panels', JSON.stringify(panels));
  }, [panels]);

  useEffect(() => {
    loadSessionHistory();
  }, []);

  const loadSessionHistory = async () => {
    try {
      const sessions = await sessionManager.listSessions?.() ?? [];
      setSessionHistory(sessions.map((s: any) => ({
        id: s.id,
        name: s.name ?? `Session ${s.id}`,
        timestamp: s.timestamp ?? Date.now(),
      })));
    } catch {
      setSessionHistory([]);
    }
  };

  const handleDragStart = (panelId: string) => {
    setDraggedPanel(panelId);
  };

  const handleDragOver = (e: React.DragEvent, targetId: string) => {
    e.preventDefault();
    if (!draggedPanel || draggedPanel === targetId) return;
    
    setPanels((prev) => {
      const draggedIdx = prev.findIndex((p) => p.id === draggedPanel);
      const targetIdx = prev.findIndex((p) => p.id === targetId);
      if (draggedIdx === -1 || targetIdx === -1) return prev;
      
      const next = [...prev];
      const [removed] = next.splice(draggedIdx, 1);
      next.splice(targetIdx, 0, removed);
      return next.map((p, i) => ({ ...p, order: i }));
    });
  };

  const handleDragEnd = () => {
    setDraggedPanel(null);
  };

  const saveLayout = () => {
    localStorage.setItem('workspace-layout', JSON.stringify(panels));
    addTimelineEvent('Layout saved');
  };

  const restoreLayout = () => {
    const saved = localStorage.getItem('workspace-layout');
    if (saved) {
      setPanels(JSON.parse(saved));
      addTimelineEvent('Layout restored');
    }
  };

  const resetLayout = () => {
    setPanels(DEFAULT_PANELS);
    localStorage.removeItem('workspace-panels');
    addTimelineEvent('Layout reset');
  };

  const togglePanel = (panelId: string) => {
    setPanels((prev) =>
      prev.map((p) => (p.id === panelId ? { ...p, visible: !p.visible } : p))
    );
  };

  const sortedPanels = [...panels].sort((a, b) => a.order - b.order);

  const renderPanel = (panel: PanelConfig) => {
    if (!panel.visible) return null;

    const isDraggedOver = draggedPanel && draggedPanel !== panel.id;

    return (
      <div
        key={panel.id}
        draggable
        onDragStart={() => handleDragStart(panel.id)}
        onDragOver={(e) => handleDragOver(e, panel.id)}
        onDragEnd={handleDragEnd}
        style={{
          ...card,
          opacity: isDraggedOver ? 0.8 : 1,
          border: isDraggedOver ? `2px dashed ${colors.primary}` : card.border,
          transition: 'all 0.15s ease',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            cursor: 'grab',
            padding: '4px 0',
          }}
        >
          <GripVertical size={14} color={colors.bodyMuted} />
          <span style={{ fontWeight: 700, color: colors.ink, flex: 1 }}>{panel.title}</span>
          <button
            onClick={() => togglePanel(panel.id)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: 4,
              color: colors.bodyMuted,
            }}
          >
            <Trash2 size={12} />
          </button>
        </div>
        {panel.id === 'stats' && renderStats()}
        {panel.id === 'workspaces' && renderWorkspaces()}
        {panel.id === 'activity' && renderActivity()}
        {panel.id === 'sessions' && renderSessionHistory()}
      </div>
    );
  };

  const renderStats = () => (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
      <div style={stat}>
        <span style={{ fontSize: 11, color: colors.bodyMuted }}>Active model</span>
        <span style={{ fontWeight: 700, color: colors.ink }}>{model.modelInfo?.model_name ?? (model.loading ? 'Loading…' : 'None')}</span>
        {model.modelInfo && <span style={{ fontSize: 11, color: colors.bodyMuted }}>{model.modelInfo.num_layers ?? '–'} layers · {model.modelInfo.num_heads ?? '–'} heads</span>}
      </div>
      <div style={stat}>
        <span style={{ fontSize: 11, color: colors.bodyMuted }}>Timeline events</span>
        <span style={{ fontWeight: 700, color: colors.ink }}>{timeline.length}</span>
      </div>
      <div style={stat}>
        <span style={{ fontSize: 11, color: colors.bodyMuted }}>Notes</span>
        <span style={{ fontWeight: 700, color: colors.ink }}>{workspaceNotes.length}</span>
      </div>
      <div style={stat}>
        <span style={{ fontSize: 11, color: colors.bodyMuted }}>Console entries</span>
        <span style={{ fontWeight: 700, color: colors.ink }}>{consoleLogs.length}</span>
      </div>
    </div>
  );

  const renderWorkspaces = () => (
    <>
      {workspaceNames.map((name) => (
        <button
          key={name}
          onClick={() => {
            setActiveWorkspace(name);
            addTimelineEvent(`Switched workspace: ${name}`);
          }}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: name === activeWorkspace ? colors.accentSoft : colors.canvas,
            border: `1px solid ${name === activeWorkspace ? colors.primary : colors.hairline}`,
            borderRadius: 6,
            padding: '8px 12px',
            fontSize: 12,
            color: colors.ink,
            cursor: 'pointer',
            textAlign: 'left',
          }}
        >
          <span>{name}</span>
          <ChevronRight size={14} color={name === activeWorkspace ? colors.primary : colors.bodyMuted} />
        </button>
      ))}
    </>
  );

  const renderActivity = () => (
    <>
      {consoleLogs.slice(-6).reverse().map((c) => (
        <div key={c.id} style={{ display: 'flex', gap: 8, fontSize: 12 }}>
          <span style={{ color: c.level === 'error' ? colors.dangerText : c.level === 'warn' ? colors.warningText : colors.bodyMuted, minWidth: 40 }}>{c.level}</span>
          <span style={{ color: colors.body, fontFamily: 'monospace', flex: 1 }}>{c.message}</span>
        </div>
      ))}
    </>
  );

  const renderSessionHistory = () => (
    <>
      {sessionHistory.length === 0 ? (
        <div style={{ fontSize: 12, color: colors.bodyMuted }}>No sessions yet</div>
      ) : (
        sessionHistory.slice(0, 5).map((session) => (
          <div
            key={session.id}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '6px 0',
              borderBottom: `1px solid ${colors.hairline}`,
            }}
          >
            <Clock size={12} color={colors.bodyMuted} />
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 12, color: colors.ink }}>{session.name}</div>
              <div style={{ fontSize: 10, color: colors.bodyMuted }}>
                {new Date(session.timestamp).toLocaleString()}
              </div>
            </div>
          </div>
        ))
      )}
    </>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <FolderTree size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Workspace</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{activeWorkspace}</span>
      </div>

      {/* Layout Controls */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button onClick={saveLayout} style={{ ...btn, background: colors.surfacePearl, color: colors.ink, border: `1px solid ${colors.hairline}` }}>
          <Save size={13} />
          Save Layout
        </button>
        <button onClick={restoreLayout} style={{ ...btn, background: colors.surfacePearl, color: colors.ink, border: `1px solid ${colors.hairline}` }}>
          <RotateCcw size={13} />
          Restore
        </button>
        <button onClick={resetLayout} style={{ ...btn, background: colors.surfacePearl, color: colors.ink, border: `1px solid ${colors.hairline}` }}>
          Reset
        </button>
      </div>

      {/* Panels */}
      {sortedPanels.map(renderPanel)}

      <div style={{ fontSize: 11, color: colors.bodyMuted }}>
        {model.loaded ? 'Model ready — run prompts from the GPT-2 Live tool.' : 'No model loaded — use the Models or GPT-2 Live toolbar.'}
      </div>
    </div>
  );
};
