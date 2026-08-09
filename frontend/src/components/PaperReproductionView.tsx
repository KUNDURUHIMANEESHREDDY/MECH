import React, { useState } from 'react';
import { FileCheck2, Play, Loader2, Trash2 } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useModel } from '../shared/hooks/useModel';
import { useWorkspaceStore } from '../shared/stores/workspace';

interface Paper {
  id: string;
  title: string;
  prompt: string;
  expect: string;
}

interface Reproduction {
  paper: string;
  runAt: string;
  passed: boolean | null;
  output: string;
  tokens: number;
}

const PAPERS: Paper[] = [
  { id: 'ioi', title: 'Indirect Object Identification (IOI)', prompt: 'When Mary and John went to the store, John gave the bag to', expect: 'Mary' },
  { id: 'induction', title: 'Induction Heads', prompt: 'The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy', expect: 'd' },
  { id: 'greater-than', title: 'Greater-Than Arithmetic', prompt: 'The number 2024 is greater than 3000? No, 2024 is', expect: '2' },
  { id: 'gpt2-component', title: 'GPT-2 Component Analysis', prompt: 'The capital of France is', expect: ' France' },
];

const STORAGE = 'mech.reproductions';

const load = (): Reproduction[] => {
  try {
    return JSON.parse(localStorage.getItem(STORAGE) ?? '[]') as Reproduction[];
  } catch {
    return [];
  }
};

export const PaperReproductionView: React.FC = () => {
  const { state: model, infer } = useModel();
  const { addTimelineEvent, addConsoleLog } = useWorkspaceStore();
  const [repros, setRepros] = useState<Reproduction[]>(load);
  const [runningId, setRunningId] = useState<string | null>(null);

  const runPaper = async (p: Paper) => {
    if (!model.loaded || runningId) return;
    setRunningId(p.id);
    addConsoleLog('info', `Reproduction started: ${p.title}`);
    try {
      const r = await infer(p.prompt, 12);
      const output = r.generatedText;
      const passed = p.expect ? output.includes(p.expect) : null;
      const entry: Reproduction = { paper: p.title, runAt: new Date().toLocaleTimeString(), passed, output, tokens: r.tokens.length };
      const next = [entry, ...repros].slice(0, 40);
      setRepros(next);
      try {
        localStorage.setItem(STORAGE, JSON.stringify(next));
      } catch {
        void 0;
      }
      addTimelineEvent(`Reproduction ${passed === null ? 'ran' : passed ? 'PASSED' : 'FAILED'}: ${p.title}`);
    } catch (err) {
      const entry: Reproduction = { paper: p.title, runAt: new Date().toLocaleTimeString(), passed: false, output: `ERR: ${(err as Error).message}`, tokens: 0 };
      setRepros((prev) => [entry, ...prev].slice(0, 40));
    }
    setRunningId(null);
  };

  const runAll = async () => {
    for (const p of PAPERS) {
      if (runningId) return;
      await runPaper(p);
    }
  };

  const clear = () => {
    setRepros([]);
    try {
      localStorage.removeItem(STORAGE);
    } catch {
      void 0;
    }
  };

  const passedCount = repros.filter((r) => r.passed === true).length;
  const failedCount = repros.filter((r) => r.passed === false).length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <FileCheck2 size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Paper Reproduction</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{passedCount} pass · {failedCount} fail</span>
        {repros.length > 0 && (
          <button onClick={clear} style={{ background: 'none', border: 'none', color: colors.dangerText, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
            <Trash2 size={12} /> Clear
          </button>
        )}
      </div>

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Canonical probe set</div>
        <div style={{ fontSize: 11, color: colors.bodyMuted }}>
          Each probe encodes a finding from a published interpretability paper. A run passes when the generated continuation contains the paper's expected subject.
        </div>
        <button
          onClick={() => void runAll()}
          disabled={!model.loaded || runningId !== null}
          style={{ alignSelf: 'flex-start', background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: !model.loaded || runningId !== null ? 0.5 : 1 }}
        >
          {runningId ? <Loader2 size={13} /> : <Play size={13} />} {runningId ? 'Running…' : 'Reproduce all'}
        </button>
        {!model.loaded && <span style={{ fontSize: 11, color: colors.bodyMuted }}>Load a model first (Models or GPT-2 toolbar)</span>}
      </div>

      {PAPERS.map((p) => (
        <div key={p.id} style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontWeight: 700, color: colors.ink }}>{p.title}</span>
            <span style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
              <span style={{ fontSize: 10, color: colors.bodyMuted }}>expects “{p.expect}”</span>
              <button
                onClick={() => void runPaper(p)}
                disabled={!model.loaded || runningId !== null}
                style={{ background: colors.accentSoft, color: colors.primary, border: `1px solid ${colors.hairline}`, borderRadius: 6, padding: '5px 10px', fontSize: 11, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 5, opacity: !model.loaded || runningId !== null ? 0.5 : 1 }}
              >
                {runningId === p.id ? <Loader2 size={12} /> : <Play size={12} />} Run
              </button>
            </span>
          </div>
          <div style={{ fontSize: 12, color: colors.bodyMuted, fontFamily: 'monospace' }}>{p.prompt}</div>
          {repros.filter((r) => r.paper === p.title)[0] && (() => {
            const latest = repros.filter((r) => r.paper === p.title)[0];
            return (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, borderTop: `1px solid ${colors.dividerSoft}`, paddingTop: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      color: latest.passed === null ? colors.bodyMuted : latest.passed ? colors.successText : colors.dangerText,
                      background: latest.passed === null ? colors.canvasParchment : latest.passed ? colors.successSoft : colors.dangerSoft,
                      borderRadius: 10,
                      padding: '2px 8px',
                    }}
                  >
                    {latest.passed === null ? 'Ran' : latest.passed ? 'PASS' : 'FAIL'}
                  </span>
                  <span style={{ fontSize: 11, color: colors.bodyMuted }}>{latest.runAt} · {latest.tokens} tok</span>
                </div>
                <div style={{ fontSize: 13, color: colors.ink, fontFamily: 'Georgia, serif' }}>{latest.output}</div>
              </div>
            );
          })()}
        </div>
      ))}
    </div>
  );
};
