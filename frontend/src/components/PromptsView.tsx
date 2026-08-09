import React, { useState } from 'react';
import { Book, Play, Loader2, Trash2, Plus } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useModel } from '../shared/hooks/useModel';
import { useWorkspaceStore } from '../shared/stores/workspace';

interface SavedPrompt {
  id: string;
  title: string;
  prompt: string;
  createdAt: string;
}

const STORAGE = 'mech.prompts';

const load = (): SavedPrompt[] => {
  try {
    return JSON.parse(localStorage.getItem(STORAGE) ?? '[]') as SavedPrompt[];
  } catch {
    return [];
  }
};

export const PromptsView: React.FC = () => {
  const { state: model, infer } = useModel();
  const { addTimelineEvent, addConsoleLog } = useWorkspaceStore();
  const [prompts, setPrompts] = useState<SavedPrompt[]>(load);
  const [title, setTitle] = useState('');
  const [promptText, setPromptText] = useState('');
  const [runningId, setRunningId] = useState<string | null>(null);
  const [outputs, setOutputs] = useState<Record<string, string>>({});

  const persist = (next: SavedPrompt[]) => {
    setPrompts(next);
    try {
      localStorage.setItem(STORAGE, JSON.stringify(next));
    } catch {
      addConsoleLog('warn', 'Prompts: localStorage unavailable, keeping in-memory only');
    }
  };

  const save = () => {
    if (!title.trim() || !promptText.trim()) return;
    const p: SavedPrompt = { id: `prm_${Date.now()}`, title: title.trim(), prompt: promptText.trim(), createdAt: new Date().toLocaleString() };
    persist([...prompts, p]);
    setTitle('');
    setPromptText('');
    addTimelineEvent(`Prompt saved: ${p.title}`);
  };

  const runPrompt = async (p: SavedPrompt) => {
    if (!model.loaded || runningId) return;
    setRunningId(p.id);
    try {
      const r = await infer(p.prompt, 12);
      setOutputs((prev) => ({ ...prev, [p.id]: r.generatedText }));
      addTimelineEvent(`Prompt run: ${p.title} (${r.tokens.length} tokens)`);
    } catch (err) {
      setOutputs((prev) => ({ ...prev, [p.id]: `ERR: ${(err as Error).message}` }));
    }
    setRunningId(null);
  };

  const removePrompt = (id: string) => {
    persist(prompts.filter((p) => p.id !== id));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Book size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Prompts Library</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{prompts.length} saved</span>
      </div>

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Title"
          style={{ padding: 6, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontSize: 12, background: colors.canvas, color: colors.ink }}
        />
        <textarea
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          rows={3}
          placeholder="Prompt text"
          style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, resize: 'vertical', background: colors.canvas, color: colors.ink }}
        />
        <button onClick={save} disabled={!title.trim() || !promptText.trim()} style={{ alignSelf: 'flex-start', background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: !title.trim() || !promptText.trim() ? 0.5 : 1 }}>
          <Plus size={13} /> Save prompt
        </button>
      </div>

      {prompts.length === 0 && (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          No saved prompts yet. Save reusable probes and run them against the loaded model.
        </div>
      )}

      {prompts.map((p) => (
        <div key={p.id} style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontWeight: 700, color: colors.ink }}>{p.title}</span>
            <span style={{ fontSize: 11, color: colors.bodyMuted }}>{p.createdAt}</span>
            <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
              <button onClick={() => void runPrompt(p)} disabled={!model.loaded || runningId !== null} style={{ background: colors.accentSoft, color: colors.primary, border: `1px solid ${colors.hairline}`, borderRadius: 6, padding: '5px 10px', fontSize: 11, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 5, opacity: !model.loaded || runningId !== null ? 0.5 : 1 }}>
                {runningId === p.id ? <Loader2 size={12} /> : <Play size={12} />} {runningId === p.id ? 'Running…' : 'Run'}
              </button>
              <button onClick={() => removePrompt(p.id)} style={{ background: 'none', border: 'none', color: colors.dangerText, cursor: 'pointer', display: 'inline-flex', alignItems: 'center' }}>
                <Trash2 size={13} />
              </button>
            </span>
          </div>
          <div style={{ fontSize: 12, color: colors.bodyMuted, fontFamily: 'monospace' }}>{p.prompt}</div>
          {outputs[p.id] && (
            <div style={{ fontSize: 13, color: colors.ink, fontFamily: 'Georgia, serif', borderTop: `1px solid ${colors.dividerSoft}`, paddingTop: 6 }}>{outputs[p.id]}</div>
          )}
        </div>
      ))}
    </div>
  );
};
