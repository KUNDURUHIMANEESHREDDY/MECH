import React, { useState } from 'react';
import { NotebookPen, Plus } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useWorkspaceStore } from '../shared/stores/workspace';

export const ResearchNotebook: React.FC = () => {
  const { notes, addNote, addConsoleLog } = useWorkspaceStore();
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');

  const save = () => {
    if (!title.trim() && !content.trim()) return;
    addNote(title.trim() || 'Untitled note', content.trim());
    addConsoleLog('info', `Notebook entry added: ${title.trim() || 'Untitled note'}`);
    setTitle('');
    setContent('');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <NotebookPen size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Research Notebook</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{notes.length} notes</span>
      </div>

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Note title"
          style={{ padding: 6, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontSize: 12, background: colors.canvas, color: colors.ink }}
        />
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          rows={4}
          placeholder="Observation, hypothesis, finding…"
          style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontSize: 12, resize: 'vertical', background: colors.canvas, color: colors.ink }}
        />
        <button onClick={save} disabled={!title.trim() && !content.trim()} style={{ alignSelf: 'flex-start', background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: !title.trim() && !content.trim() ? 0.5 : 1 }}>
          <Plus size={13} /> Add note
        </button>
      </div>

      {notes.length === 0 && (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          No notes yet. Notes are shared with the workspace store and persist with saved sessions.
        </div>
      )}

      {notes
        .slice()
        .reverse()
        .map((n) => (
          <div key={n.id} style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ fontWeight: 700, color: colors.ink }}>{n.title}</div>
            <div style={{ fontSize: 12, color: colors.body, lineHeight: 1.5 }}>{n.content}</div>
          </div>
        ))}
    </div>
  );
};
