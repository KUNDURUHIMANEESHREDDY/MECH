import React, { useState } from 'react';
import { FlaskConical, Trash2 } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { ExperimentRun } from './ExperimentsView';

const STORAGE = 'mech.experiments';

const load = (): ExperimentRun[] => {
  try {
    return JSON.parse(localStorage.getItem(STORAGE) ?? '[]') as ExperimentRun[];
  } catch {
    return [];
  }
};

export const ExperimentNotebook: React.FC = () => {
  const [runs, setRuns] = useState<ExperimentRun[]>(load);

  const clear = () => {
    setRuns([]);
    try {
      localStorage.removeItem(STORAGE);
    } catch {
      void 0;
    }
  };

  const okCount = runs.filter((r) => r.ok).length;
  const totalTokens = runs.reduce((a, r) => a + r.tokens, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <FlaskConical size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Experiment Notebook</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>
          {runs.length} runs · {okCount} ok · {totalTokens} tok
        </span>
        {runs.length > 0 && (
          <button onClick={clear} style={{ background: 'none', border: 'none', color: colors.dangerText, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
            <Trash2 size={12} /> Clear
          </button>
        )}
      </div>

      {runs.length === 0 && (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          No records yet — runs from the Experiments tool appear here as immutable notebook entries.
        </div>
      )}

      {runs
        .slice()
        .reverse()
        .map((r) => (
          <div key={r.name} style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ width: 8, height: 8, borderRadius: 4, background: r.ok ? colors.success : colors.dangerText, flexShrink: 0 }} />
              <span style={{ fontSize: 11, color: colors.bodyMuted, fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{r.prompt}</span>
              <span style={{ fontSize: 11, color: colors.bodyMuted, whiteSpace: 'nowrap' }}>{r.tokens} tok · {r.latency} ms</span>
            </div>
            <div style={{ fontSize: 13, color: r.ok ? colors.body : colors.dangerText, fontFamily: 'Georgia, serif', lineHeight: 1.5 }}>{r.output}</div>
            <div style={{ fontSize: 10, color: colors.bodyMuted }}>{r.model}</div>
          </div>
        ))}
    </div>
  );
};
