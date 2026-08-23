import React, { useState, useEffect } from 'react';
import { FileCheck2, Play, Loader2, Trash2, RefreshCw } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useModel } from '../shared/hooks/useModel';
import { useWorkspaceStore } from '../shared/stores/workspace';
import { api } from '../services/api';

interface Paper {
  id: string;
  title: string;
  prompt: string;
  expect: string;
  category?: string;
}

interface Reproduction {
  paper: string;
  runAt: string;
  passed: boolean | null;
  output: string;
  tokens: number;
}

// Fallback seed pools if API is initializing
const INITIAL_PAPERS: Paper[] = [
  { id: 'ioi', title: 'Indirect Object Identification (IOI)', prompt: 'When Mary gave the book to John, John thanked', expect: 'Mary', category: 'ioi' },
  { id: 'induction', title: 'Induction Heads', prompt: 'The sequence a b a b a b a', expect: 'b', category: 'induction' },
  { id: 'arithmetic', title: 'Arithmetic & Numeric Completion', prompt: 'Two plus two equals', expect: 'four', category: 'arithmetic' },
  { id: 'factual', title: 'Factual Recall & World Knowledge', prompt: 'The capital of France is', expect: 'Paris', category: 'factual_recall' },
  { id: 'relational', title: 'Relational & Predicate Binding', prompt: 'The official language of Spain is', expect: 'Spanish', category: 'relational' },
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
  const [papers, setPapers] = useState<Paper[]>(INITIAL_PAPERS);
  const [repros, setRepros] = useState<Reproduction[]>(load);
  const [runningId, setRunningId] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState<boolean>(false);

  const fetchDynamicProbes = async () => {
    setRefreshing(true);
    try {
      const res = await api.getSessionProbes();
      if (res && Array.isArray(res.probes) && res.probes.length > 0) {
        const mapped: Paper[] = res.probes.slice(0, 6).map((p) => ({
          id: p.probe_id,
          title: formatCategoryTitle(p.category),
          prompt: p.clean_prompt,
          expect: p.target_token.trim(),
          category: p.category,
        }));
        setPapers(mapped);
        addConsoleLog('info', `Dynamic session probes loaded (${mapped.length} prompts).`);
      }
    } catch {
      // Fallback remains active
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    void fetchDynamicProbes();
  }, []);

  const formatCategoryTitle = (cat: string) => {
    switch (cat) {
      case 'ioi':
        return 'Indirect Object Identification (IOI)';
      case 'induction':
        return 'Induction Head Pattern';
      case 'arithmetic':
        return 'Arithmetic Reasoning';
      case 'factual_recall':
        return 'Factual Recall Probe';
      case 'relational':
        return 'Relational Binding Probe';
      default:
        return cat.replace('_', ' ').toUpperCase();
    }
  };

  const runPaper = async (p: Paper) => {
    if (!model.loaded || runningId) return;
    setRunningId(p.id);
    addConsoleLog('info', `Reproduction started: ${p.title}`);
    try {
      const r = await infer(p.prompt, 12);
      const output = r.generatedText;
      const passed = p.expect ? output.toLowerCase().includes(p.expect.toLowerCase()) : null;
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
    for (const p of papers) {
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
        <button
          onClick={() => void fetchDynamicProbes()}
          disabled={refreshing}
          style={{ background: 'none', border: `1px solid ${colors.hairline}`, borderRadius: 4, padding: '2px 8px', color: colors.ink, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}
        >
          <RefreshCw size={11} className={refreshing ? 'spin' : ''} /> Refresh Prompts
        </button>
        {repros.length > 0 && (
          <button onClick={clear} style={{ background: 'none', border: 'none', color: colors.dangerText, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
            <Trash2 size={12} /> Clear
          </button>
        )}
      </div>

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Dynamic Session Probes</div>
        <div style={{ fontSize: 11, color: colors.bodyMuted }}>
          Prompts are dynamically generated at app open time across IOI, Induction, Arithmetic, Factual Recall, and Relational Binding categories.
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

      {papers.map((p) => (
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
