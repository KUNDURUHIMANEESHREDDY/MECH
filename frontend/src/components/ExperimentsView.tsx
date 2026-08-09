import React, { useState } from 'react';
import { FlaskConical, Play, Loader2, RotateCcw } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useModel } from '../shared/hooks/useModel';
import { useWorkspaceStore } from '../shared/stores/workspace';

export interface ExperimentRun {
  name: string;
  model: string;
  prompt: string;
  tokens: number;
  latency: number;
  output: string;
  ok: boolean;
}

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

const STORAGE = 'mech.experiments';

export const ExperimentsView: React.FC = () => {
  const { state: model, infer } = useModel();
  const { addTimelineEvent, addConsoleLog } = useWorkspaceStore();
  const [input, setInput] = useState('The capital of France is\nWhen Mary and John went to the store, John gave the bag to');
  const [runs, setRuns] = useState<ExperimentRun[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE) ?? '[]') as ExperimentRun[];
    } catch {
      return [];
    }
  });
  const [running, setRunning] = useState(false);
  const [current, setCurrent] = useState<string | null>(null);

  const persist = (next: ExperimentRun[]) => {
    setRuns(next);
    try {
      localStorage.setItem(STORAGE, JSON.stringify(next.slice(-50)));
    } catch {
      addConsoleLog('warn', 'Experiments: localStorage unavailable, keeping in-memory history');
    }
  };

  const runAll = async () => {
    const prompts = input.split('\n').map((p) => p.trim()).filter(Boolean);
    if (!prompts.length || !model.loaded) return;
    setRunning(true);
    const out: ExperimentRun[] = [];
    for (const p of prompts) {
      setCurrent(p.slice(0, 40));
      const t0 = performance.now();
      try {
        const r = await infer(p, 8);
        out.push({
          name: `run_${Date.now()}_${out.length}`,
          model: model.modelInfo?.model_name ?? 'unknown',
          prompt: p,
          tokens: r.tokens.length,
          latency: Math.round(performance.now() - t0),
          output: r.generatedText,
          ok: true,
        });
        addTimelineEvent(`Experiment completed: ${p.slice(0, 30)} (${r.tokens.length} tokens)`);
      } catch (e) {
        out.push({ name: `err_${Date.now()}_${out.length}`, model: model.modelInfo?.model_name ?? 'unknown', prompt: p, tokens: 0, latency: 0, output: `ERR: ${(e as Error).message}`, ok: false });
      }
    }
    persist([...runs, ...out]);
    setRunning(false);
    setCurrent(null);
  };

  const clearHistory = () => {
    setRuns([]);
    try {
      localStorage.removeItem(STORAGE);
    } catch {
      addConsoleLog('warn', 'Experiments history could not be cleared from storage');
    }
  };

  const avgTokens = runs.length ? Math.round(runs.reduce((a, r) => a + r.tokens, 0) / runs.length) : 0;
  const okCount = runs.filter((r) => r.ok).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <FlaskConical size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Experiments</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{runs.length} runs · {okCount} ok · avg {avgTokens} tok</span>
      </div>

      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Prompts (one per line)</div>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={5}
          style={{ width: '100%', boxSizing: 'border-box', padding: 8, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, resize: 'vertical', background: colors.canvas, color: colors.ink }}
        />
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <button onClick={runAll} disabled={!model.loaded || running} style={{ ...btn, opacity: !model.loaded || running ? 0.5 : 1 }}>
            {running ? <Loader2 size={13} /> : <Play size={13} />}
            {running && current ? `Running: ${current}…` : 'Run batch'}
          </button>
          {runs.length > 0 && (
            <button onClick={clearHistory} style={{ background: 'none', border: 'none', color: colors.bodyMuted, fontSize: 11, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <RotateCcw size={12} /> Clear history
            </button>
          )}
          {!model.loaded && <span style={{ fontSize: 11, color: colors.bodyMuted }}>Load a model first (Models or GPT-2 toolbar)</span>}
        </div>
      </div>

      {runs.length === 0 ? (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          No runs yet. Paste prompts and run a batch — results are saved to this session.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', background: colors.canvas, borderRadius: 8 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', fontSize: 10, textTransform: 'uppercase', color: colors.bodyMuted, padding: '4px 8px', borderBottom: `1px solid ${colors.hairline}` }}>Prompt</th>
                <th style={{ textAlign: 'left', fontSize: 10, textTransform: 'uppercase', color: colors.bodyMuted, padding: '4px 8px', borderBottom: `1px solid ${colors.hairline}` }}>Tokens</th>
                <th style={{ textAlign: 'left', fontSize: 10, textTransform: 'uppercase', color: colors.bodyMuted, padding: '4px 8px', borderBottom: `1px solid ${colors.hairline}` }}>Latency</th>
                <th style={{ textAlign: 'left', fontSize: 10, textTransform: 'uppercase', color: colors.bodyMuted, padding: '4px 8px', borderBottom: `1px solid ${colors.hairline}` }}>Output</th>
              </tr>
            </thead>
            <tbody>
              {runs.slice(-25).reverse().map((r) => (
                <tr key={r.name}>
                  <td style={{ fontSize: 12, padding: '6px 8px', borderBottom: `1px solid ${colors.dividerSoft}`, color: colors.ink, fontFamily: 'monospace', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.prompt}</td>
                  <td style={{ fontSize: 12, padding: '6px 8px', borderBottom: `1px solid ${colors.dividerSoft}`, color: colors.body, fontVariantNumeric: 'tabular-nums' }}>{r.tokens}</td>
                  <td style={{ fontSize: 12, padding: '6px 8px', borderBottom: `1px solid ${colors.dividerSoft}`, color: colors.body, fontVariantNumeric: 'tabular-nums' }}>{r.latency} ms</td>
                  <td style={{ fontSize: 12, padding: '6px 8px', borderBottom: `1px solid ${colors.dividerSoft}`, color: r.ok ? colors.body : colors.dangerText, fontFamily: 'Georgia, serif', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.output}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
