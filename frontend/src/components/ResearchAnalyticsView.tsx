import React, { useState } from 'react';
import { BarChart3, RefreshCw } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { ExperimentRun } from './ExperimentsView';

const myStorage = (key: string) => {
  try {
    return JSON.parse(localStorage.getItem(key) ?? '[]');
  } catch {
    return [];
  }
};

export const ResearchAnalyticsView: React.FC = () => {
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);
  const runs = (myStorage('mech.experiments') as ExperimentRun[]).slice(-20);
  const reports = myStorage('mech.reports') as unknown[];

  const totalTokens = runs.reduce((a, r) => a + r.tokens, 0);
  const okRuns = runs.filter((r) => r.ok).length;
  const avgLatency = runs.length ? Math.round(runs.reduce((a, r) => a + r.latency, 0) / runs.length) : 0;
  const maxTokens = Math.max(...runs.map((r) => r.tokens), 1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <BarChart3 size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Analytics</span>
        <button onClick={() => setLastRefresh(new Date().toLocaleTimeString())} style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
          <RefreshCw size={12} /> Refresh
        </button>
        {lastRefresh && <span style={{ fontSize: 11, color: colors.bodyMuted }}>{lastRefresh}</span>}
      </div>

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {[
          { label: 'Experiment runs', value: String(runs.length) },
          { label: 'Successful runs', value: `${okRuns}/${runs.length || 0}` },
          { label: 'Tokens generated', value: String(totalTokens) },
          { label: 'Avg latency', value: runs.length ? `${avgLatency} ms` : '–' },
          { label: 'Reports saved', value: String(reports.length) },
        ].map((s) => (
          <div key={s.label} style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 4, flex: 1, minWidth: 130 }}>
            <span style={{ fontSize: 11, color: colors.bodyMuted }}>{s.label}</span>
            <span style={{ fontWeight: 700, fontSize: 18, color: colors.ink, fontVariantNumeric: 'tabular-nums' }}>{s.value}</span>
          </div>
        ))}
      </div>

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12 }}>
        <div style={{ fontWeight: 700, color: colors.ink, marginBottom: 10 }}>Tokens per run (last {Math.min(runs.length, 20) || 0})</div>
        {runs.length === 0 ? (
          <div style={{ fontSize: 12, color: colors.bodyMuted }}>No experiment data yet — run a batch from the Experiments tool.</div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 100 }}>
            {runs.slice(-20).map((r, i) => (
              <div
                key={i}
                title={`${r.prompt.slice(0, 30)} — ${r.tokens} tok`}
                style={{
                  width: 18,
                  height: Math.max(4, Math.round((r.tokens / maxTokens) * 96)),
                  background: r.ok ? colors.primary : colors.dangerText,
                  borderRadius: '2px 2px 0 0',
                }}
              />
            ))}
          </div>
        )}
        <div style={{ fontSize: 11, color: colors.bodyMuted, marginTop: 6 }}>Bar height = generated tokens per run; red = failed run.</div>
      </div>
    </div>
  );
};
