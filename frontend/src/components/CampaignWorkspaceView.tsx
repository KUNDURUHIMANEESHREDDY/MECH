import React, { useState } from 'react';
import { Megaphone, Play, Loader2, Trash2, Plus } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useModel } from '../shared/hooks/useModel';
import { useWorkspaceStore } from '../shared/stores/workspace';

interface Campaign {
  id: string;
  name: string;
  prompts: string[];
  createdAt: string;
}

interface CampaignResult {
  prompt: string;
  ok: boolean;
  output: string;
  tokens: number;
}

const STORAGE = 'mech.campaigns';

const load = (): Campaign[] => {
  try {
    return JSON.parse(localStorage.getItem(STORAGE) ?? '[]') as Campaign[];
  } catch {
    return [];
  }
};

export const CampaignWorkspaceView: React.FC = () => {
  const { state: model, infer } = useModel();
  const { addTimelineEvent, addConsoleLog } = useWorkspaceStore();
  const [campaigns, setCampaigns] = useState<Campaign[]>(load);
  const [name, setName] = useState('');
  const [promptText, setPromptText] = useState('The capital of France is\nWhen Mary and John went to the store, John gave the bag to');
  const [runningId, setRunningId] = useState<string | null>(null);
  const [results, setResults] = useState<Record<string, CampaignResult[]>>({});

  const persist = (next: Campaign[]) => {
    setCampaigns(next);
    try {
      localStorage.setItem(STORAGE, JSON.stringify(next));
    } catch {
      addConsoleLog('warn', 'Campaigns: localStorage unavailable, keeping in-memory only');
    }
  };

  const createCampaign = () => {
    const prompts = promptText.split('\n').map((p) => p.trim()).filter(Boolean);
    if (!name.trim() || !prompts.length) return;
    const c: Campaign = { id: `camp_${Date.now()}`, name: name.trim(), prompts, createdAt: new Date().toLocaleString() };
    persist([...campaigns, c]);
    setName('');
    addTimelineEvent(`Campaign created: ${c.name} (${prompts.length} prompts)`);
  };

  const runCampaign = async (c: Campaign) => {
    if (!model.loaded || runningId) return;
    setRunningId(c.id);
    addConsoleLog('info', `Campaign started: ${c.name}`);
    const out: CampaignResult[] = [];
    for (const p of c.prompts) {
      try {
        const r = await infer(p, 8);
        out.push({ prompt: p, ok: true, output: r.generatedText, tokens: r.tokens.length });
      } catch (err) {
        out.push({ prompt: p, ok: false, output: `ERR: ${(err as Error).message}`, tokens: 0 });
      }
    }
    setResults((prev) => ({ ...prev, [c.id]: out }));
    addConsoleLog('info', `Campaign finished: ${c.name} (${out.filter((o) => o.ok).length}/${out.length} ok)`);
    setRunningId(null);
  };

  const removeCampaign = (id: string) => {
    setResults((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
    persist(campaigns.filter((c) => c.id !== id));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Megaphone size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Campaigns</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{campaigns.length} campaigns</span>
      </div>

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ fontWeight: 700, color: colors.ink }}>New campaign</div>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Campaign name"
          style={{ padding: 6, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontSize: 12, background: colors.canvas, color: colors.ink }}
        />
        <textarea
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          rows={4}
          placeholder={'One prompt per line'}
          style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, resize: 'vertical', background: colors.canvas, color: colors.ink }}
        />
        <button
          onClick={createCampaign}
          disabled={!name.trim() || !promptText.trim()}
          style={{ alignSelf: 'flex-start', background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: !name.trim() || !promptText.trim() ? 0.5 : 1 }}
        >
          <Plus size={13} /> Save campaign
        </button>
      </div>

      {campaigns.length === 0 && (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          No campaigns yet — batch prompts are run through the loaded model sequentially.
        </div>
      )}

      {campaigns.map((c) => (
        <div key={c.id} style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontWeight: 700, color: colors.ink }}>{c.name}</span>
            <span style={{ fontSize: 11, color: colors.bodyMuted }}>{c.prompts.length} prompts · {c.createdAt}</span>
            <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
              <button
                onClick={() => void runCampaign(c)}
                disabled={!model.loaded || runningId !== null}
                style={{ background: colors.accentSoft, color: colors.primary, border: `1px solid ${colors.hairline}`, borderRadius: 6, padding: '5px 10px', fontSize: 11, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 5, opacity: !model.loaded || runningId !== null ? 0.5 : 1 }}
              >
                {runningId === c.id ? <Loader2 size={12} /> : <Play size={12} />} {runningId === c.id ? 'Running…' : 'Run'}
              </button>
              <button onClick={() => removeCampaign(c.id)} style={{ background: 'none', border: 'none', color: colors.dangerText, cursor: 'pointer', display: 'inline-flex', alignItems: 'center' }}>
                <Trash2 size={13} />
              </button>
            </span>
          </div>
          <div style={{ fontSize: 11, color: colors.bodyMuted, fontFamily: 'monospace', maxHeight: 80, overflow: 'auto' }}>
            {c.prompts.map((p, i) => (
              <div key={i}>{i + 1}. {p}</div>
            ))}
          </div>
          {results[c.id] && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {results[c.id].map((r, i) => (
                <div key={i} style={{ borderLeft: `3px solid ${r.ok ? colors.success : colors.dangerText}`, paddingLeft: 8, background: colors.canvasParchment, borderRadius: 4 }}>
                  <div style={{ fontSize: 11, color: colors.ink, fontFamily: 'monospace' }}>{r.prompt}</div>
                  <div style={{ fontSize: 12, color: r.ok ? colors.body : colors.dangerText, fontFamily: 'Georgia, serif' }}>{r.output}</div>
                  <div style={{ fontSize: 10, color: colors.bodyMuted }}>{r.tokens} tokens {r.ok ? '' : '(failed)'}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};
