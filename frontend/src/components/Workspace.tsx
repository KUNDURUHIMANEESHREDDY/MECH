import React from 'react';
import { FolderTree, Activity, Library, FileText, ChevronRight } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useModel } from '../shared/hooks/useModel';
import { useWorkspaceStore } from '../shared/stores/workspace';

const card: React.CSSProperties = {
  border: `1px solid ${colors.hairline}`,
  borderRadius: 8,
  background: colors.canvas,
  padding: 12,
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
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

export const Workspace: React.FC = () => {
  const { state: model } = useModel();
  const { activeWorkspace, workspaceNames, setActiveWorkspace, timeline, notes: workspaceNotes, console: consoleLogs, addTimelineEvent } = useWorkspaceStore();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <FolderTree size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Workspace</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{activeWorkspace}</span>
      </div>

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

      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Workspaces</div>
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
      </div>

      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Recent activity</div>
        {consoleLogs.slice(-6).reverse().map((c) => (
          <div key={c.id} style={{ display: 'flex', gap: 8, fontSize: 12 }}>
            <span style={{ color: c.level === 'error' ? colors.dangerText : c.level === 'warn' ? colors.warningText : colors.bodyMuted, minWidth: 40 }}>{c.level}</span>
            <span style={{ color: colors.body, fontFamily: 'monospace', flex: 1 }}>{c.message}</span>
          </div>
        ))}
      </div>

      <div style={{ fontSize: 11, color: colors.bodyMuted }}>
        {model.loaded ? 'Model ready — run prompts from the GPT-2 Live tool.' : 'No model loaded — use the Models or GPT-2 Live toolbar.'}
      </div>
    </div>
  );
};
