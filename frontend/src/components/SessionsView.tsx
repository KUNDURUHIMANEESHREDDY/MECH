import React, { useState } from 'react';
import { FolderOpen, Save, RotateCcw, Check } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useWorkspaceStore } from '../shared/stores/workspace';

export const SessionsView: React.FC = () => {
  const { workspaceNames, activeWorkspace, activeSessionId, setActiveWorkspace, saveSession, restoreSession, addConsoleLog } = useWorkspaceStore();
  const [busy, setBusy] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  const handleSave = async () => {
    setBusy('save');
    await saveSession();
    addConsoleLog('info', `Session saved for workspace: ${activeWorkspace}`);
    setSavedAt(new Date().toLocaleTimeString());
    setBusy(null);
  };

  const handleRestore = async () => {
    setBusy('restore');
    await restoreSession(activeWorkspace);
    addConsoleLog('info', `Session restored for workspace: ${activeWorkspace}`);
    setBusy(null);
  };

  const switchWorkspace = (name: string) => {
    setActiveWorkspace(name);
    addConsoleLog('info', `Switched workspace to: ${name}`);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Save size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Saved Sessions</span>
      </div>

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Active session</div>
        <div style={{ fontSize: 12, color: colors.bodyMuted }}>Workspace: <b style={{ color: colors.ink }}>{activeWorkspace}</b></div>
        <div style={{ fontSize: 12, color: colors.bodyMuted }}>Session id: <b style={{ color: colors.ink }}>{activeSessionId ?? 'none'}</b></div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button onClick={() => void handleSave()} disabled={busy !== null} style={{ background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: busy !== null ? 0.6 : 1 }}>
            <Save size={13} /> {busy === 'save' ? 'Saving…' : 'Save session'}
          </button>
          <button onClick={() => void handleRestore()} disabled={busy !== null} style={{ background: colors.accentSoft, color: colors.primary, border: `1px solid ${colors.hairline}`, borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: busy !== null ? 0.6 : 1 }}>
            <RotateCcw size={13} /> {busy === 'restore' ? 'Restoring…' : 'Restore session'}
          </button>
          {savedAt && (
            <span style={{ fontSize: 11, color: colors.success, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <Check size={12} /> saved {savedAt}
            </span>
          )}
        </div>
      </div>

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Workspaces ({workspaceNames.length})</div>
        {workspaceNames.map((w) => (
          <div key={w} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 0', borderBottom: `1px solid ${colors.dividerSoft}` }}>
            <FolderOpen size={14} color={w === activeWorkspace ? colors.primary : colors.bodyMuted} />
            <span style={{ color: w === activeWorkspace ? colors.primary : colors.ink, fontWeight: w === activeWorkspace ? 700 : 400, flex: 1 }}>{w}</span>
            {w === activeWorkspace ? (
              <span style={{ fontSize: 10, textTransform: 'uppercase', color: colors.primary, background: colors.accentSoft, borderRadius: 10, padding: '2px 8px' }}>active</span>
            ) : (
              <button onClick={() => switchWorkspace(w)} style={{ background: 'none', border: `1px solid ${colors.hairline}`, borderRadius: 6, color: colors.bodyMuted, fontSize: 11, padding: '3px 10px', cursor: 'pointer' }}>
                Switch
              </button>
            )}
          </div>
        ))}
      </div>

      <div style={{ fontSize: 11, color: colors.bodyMuted }}>
        Sessions persist panels, notes, timeline and console state per workspace. Use the JSON bridge store in the Electron main process.
      </div>
    </div>
  );
};
