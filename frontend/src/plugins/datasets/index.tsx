import React, { useState } from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { useModel } from '../../shared/hooks/useModel';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import { colors } from '../../design/tokens/colors';

interface DatasetEntry {
  id: string;
  name: string;
  shots: number;
  description: string;
  prompts: string[];
}

/**
 * Curated probing dataset catalog: every prompt is directly compatible with
 * the local GPT-2 runtime, covering the interpretability canon exercised in
 * this lab (induction, IOI, factual recall, next-token probing).
 */
const DATASETS: DatasetEntry[] = [
  {
    id: 'induction',
    name: 'Induction (copying)',
    shots: 3,
    description: 'Repeated patterns: the model must copy the first segment once it repeats. Exercises prefix-matching / QK-composition circuits.',
    prompts: [
      'The cat chased the dog. The cat chased the',
      'Alpha beta gamma alpha beta',
      'London Paris Berlin London Paris',
    ],
  },
  {
    id: 'ioi',
    name: 'Indirect Object Identification',
    shots: 3,
    description: 'Classic name-mixing probes: when John and Mary both appear, identify who performed the verb.',
    prompts: [
      'When John and Mary went to the store, Mary gave the book to',
      'After Bob and Alice met, Alice hugged',
      'Tom and Jerry argued, but Tom apologized to',
    ],
  },
  {
    id: 'factual',
    name: 'Factual recall',
    shots: 3,
    description: 'Common-knowledge completions probing stored factual associations.',
    prompts: [
      'The capital of France is',
      'Water boils at',
      'The Sun rises in the',
    ],
  },
  {
    id: 'semantic',
    name: 'Semantic relations',
    shots: 3,
    description: 'Paraphrase-prone completions to inspect synonym handling in residual stream.',
    prompts: [
      'The old man walked slowly and felt very',
      'She was cold, so she put on a warm',
      'The dog barked loudly at the',
    ],
  },
  {
    id: 'syntax',
    name: 'Syntax / agreement',
    shots: 4,
    description: 'Number and tense agreement probes for parser-like circuit behavior.',
    prompts: [
      'The children play with their',
      'He has been working for hours and has',
      'Every student brings their own',
    ],
  },
];

const DatasetViewerBody: FC<PanelContext> = () => {
  const { infer, state } = useModel();
  const workspace = useWorkspaceStore();
  const [datasets, setDatasets] = useState<DatasetEntry[]>(DATASETS);
  const [runningId, setRunningId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  React.useEffect(() => {
    void (async () => {
      try {
        const { api } = await import('../../services/api');
        const res = await api.getSessionProbes();
        if (res?.probes?.length) {
          const categorized: Record<string, string[]> = {};
          for (const p of res.probes) {
            if (!categorized[p.category]) categorized[p.category] = [];
            categorized[p.category].push(p.clean_prompt);
          }
          setDatasets((prev) =>
            prev.map((ds) => {
              if (categorized[ds.id] && categorized[ds.id].length > 0) {
                return { ...ds, prompts: categorized[ds.id], shots: categorized[ds.id].length };
              }
              return ds;
            })
          );
        }
      } catch { }
    })();
  }, []);

  const runPrompt = async (datasetId: string, prompt: string) => {
    setRunningId(datasetId);
    setStatus(`Running: "${prompt}"`);
    try {
      await infer(prompt, 12);
      workspace.addConsoleLog('info', `Dataset probe ran: ${prompt}`);
      workspace.addTimelineEvent('dataset.probe', { prompt });
      workspace.openPanel('token_viewer');
      workspace.openPanel('prediction_inspector');
    } catch (e: any) {
      setStatus(e.message || 'Inference failed.');
      workspace.addConsoleLog('error', `Dataset probe failed: ${e.message}`);
    } finally {
      setRunningId(null);
    }
  };

  return (
    <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 12, overflow: 'auto', height: '100%', backgroundColor: colors.canvas }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4 }}>
        Dataset Catalog — probing corpora
      </div>

      {status && (
        <div style={{ padding: '8px 10px', borderRadius: 6, backgroundColor: colors.infoSoft, border: `1px solid ${colors.infoBorder}`, color: colors.infoText, fontSize: 12 }}>
          {status}
          {state.running && ' (inference in progress…)'}
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {datasets.map((ds) => (
          <div key={ds.id} style={{ border: `1px solid ${colors.border}`, borderRadius: 8, padding: 12, backgroundColor: colors.surfaceTile1 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>{ds.name}</span>
              <span style={{ fontSize: 11, color: colors.bodyMuted, backgroundColor: colors.surfacePearl, borderRadius: 999, padding: '2px 8px' }}>
                {ds.shots} probes
              </span>
            </div>
            <div style={{ fontSize: 12, color: colors.bodyMuted, marginBottom: 10, lineHeight: 1.45 }}>{ds.description}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {ds.prompts.map((p) => (
                <div key={p} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <code style={{
                    flex: 1, fontSize: 12, color: colors.body, backgroundColor: colors.canvasParchment,
                    border: `1px solid ${colors.hairline}`, borderRadius: 6, padding: '6px 9px',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {p}
                  </code>
                  <button
                    onClick={() => runPrompt(ds.id, p)}
                    disabled={runningId === ds.id || state.running}
                    style={{
                      padding: '6px 12px', borderRadius: 6, border: 'none', whiteSpace: 'nowrap',
                      fontSize: 12, fontWeight: 600,
                      backgroundColor: runningId === ds.id ? colors.bodyMuted : colors.primary,
                      color: colors.onPrimary, cursor: runningId === ds.id || state.running ? 'default' : 'pointer',
                    }}
                  >
                    {runningId === ds.id ? 'Running…' : 'Run'}
                  </button>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

pluginRegistry.register({
  id: 'dataset_viewer',
  title: 'Dataset Viewer',
  icon: 'Database',
  category: 'datasets',
  resourceKinds: ['dataset'],
  defaultDock: 'center',
  Body: DatasetViewerBody,
});