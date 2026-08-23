import React, { useEffect, useState } from 'react';
import { Bug, Play, Loader2, Trash2, AlertTriangle, Cpu, CheckCircle2 } from 'lucide-react';
import { colors } from '../../design/tokens/colors';
import { useModel } from '../../shared/hooks/useModel';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import type { FC, PanelContext } from '../../shared/types';

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
const ghost: React.CSSProperties = {
  ...btn,
  background: colors.surfacePearl,
  color: colors.ink,
  border: `1px solid ${colors.hairline}`,
};

/**
 * Debugger: live view of the model pipeline (load/running/error state),
 * workspace console log and timeline, plus quick load + probe-run controls.
 */
export const DebuggerPanel: FC<PanelContext> = () => {
  const { state: model, listModels, load, infer, clearError } = useModel();
  const consoleLogs = useWorkspaceStore((s) => s.console);
  const DYNAMIC_SEEDS = [
    'The capital of France is',
    'Two plus two equals',
    'When Mary gave the book to John, John thanked',
    'The official language of Spain is',
    'The sequence a b a b a b a',
  ];
  const [probe, setProbe] = useState(() => DYNAMIC_SEEDS[Math.floor(Math.random() * DYNAMIC_SEEDS.length)]);

  useEffect(() => {
    if (model.availableModels.length === 0 && !model.loading) void listModels();
    void (async () => {
      try {
        const { api } = await import('../../services/api');
        const res = await api.getSessionProbes();
        if (res?.probes?.length) {
          setProbe(res.probes[Math.floor(Math.random() * res.probes.length)].clean_prompt);
        }
      } catch {
        // Fallback seed already set
      }
    })();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const runProbe = async () => {
    clearError();
    await infer(probe);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body }}>
      {/* State rows */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8 }}>
        {[
          { label: 'Model', value: model.modelInfo?.model_name ?? '—', tone: model.loaded ? colors.success : colors.bodyMuted, icon: <Cpu size={12} /> },
          { label: 'Loaded', value: model.loaded ? 'true' : 'false', tone: model.loaded ? colors.success : colors.danger, icon: model.loaded ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} /> },
          { label: 'Loading', value: String(model.loading), tone: model.loading ? colors.warning : colors.bodyMuted, icon: <Loader2 size={12} /> },
          { label: 'Running', value: String(model.running), tone: model.running ? colors.warning : colors.bodyMuted, icon: <Loader2 size={12} /> },
          { label: 'Layers', value: String(model.modelInfo?.num_layers ?? '—'), tone: colors.bodyMuted, icon: <Cpu size={12} /> },
          { label: 'Heads', value: String(model.modelInfo?.num_heads ?? '—'), tone: colors.bodyMuted, icon: <Cpu size={12} /> },
          { label: 'Tokens', value: String(model.result?.tokens.length ?? 0), tone: colors.bodyMuted, icon: <Cpu size={12} /> },
          { label: 'Source', value: model.modelInfo?.status === 'demo' ? 'demo' : model.modelInfo?.status ?? '—', tone: model.modelInfo?.status === 'demo' ? colors.warningText : colors.bodyMuted, icon: <Cpu size={12} /> },
        ].map((row) => (
          <div key={row.label} style={{ ...card, padding: 10 }}>
            <div style={{ fontSize: 10, color: colors.bodyMuted, textTransform: 'uppercase', letterSpacing: 0.4, display: 'flex', gap: 4, alignItems: 'center' }}>
              {row.icon} {row.label}
            </div>
            <div style={{ fontSize: 16, fontWeight: 700, color: row.tone, fontVariantNumeric: 'tabular-nums' }}>{row.value}</div>
          </div>
        ))}
      </div>

      {model.error && (
        <div style={{ background: colors.dangerSoft, color: colors.dangerText, border: `1px solid ${colors.dangerBorder}`, borderRadius: 8, padding: '8px 12px', display: 'flex', gap: 8, alignItems: 'center', fontSize: 12 }}>
          <AlertTriangle size={14} /> {model.error}
        </div>
      )}

      {/* Controls */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <button onClick={() => load(model.availableModels[0] ?? 'gpt2')} disabled={model.loading} style={{ ...ghost, opacity: model.loading ? 0.5 : 1 }}>
          {model.loading ? <Loader2 size={13} /> : null} Load default model
        </button>
        <textarea
          value={probe}
          onChange={(e) => setProbe(e.target.value)}
          rows={1}
          style={{ flex: 1, minWidth: 180, padding: 7, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontFamily: 'monospace', fontSize: 12, background: colors.canvas, color: colors.ink }}
        />
        <button onClick={runProbe} disabled={!model.loaded || model.running} style={{ ...btn, opacity: !model.loaded || model.running ? 0.5 : 1 }}>
          {model.running ? <Loader2 size={13} /> : <Play size={13} />} Run probe
        </button>
        <button onClick={clearError} style={ghost}>
          <Trash2 size={13} /> Clear error
        </button>
      </div>

      {/* Console log */}
      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Console ({consoleLogs.length})</div>
        <div style={{ maxHeight: 180, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2 }}>
          {consoleLogs.length === 0 && <div style={{ fontSize: 12, color: colors.bodyMuted }}>No log entries yet.</div>}
          {consoleLogs.map((l) => (
            <div key={l.id} style={{ display: 'flex', gap: 8, fontSize: 11, alignItems: 'baseline' }}>
              <span style={{ color: colors.inkMuted48, fontVariantNumeric: 'tabular-nums' }}>{l.ts}</span>
              <span
                style={{
                  color: l.level === 'error' ? colors.dangerText : l.level === 'warn' ? colors.warningText : colors.successText,
                  fontStyle: l.level === 'error' ? 'italic' : undefined,
                }}
              >
                [{l.level}]
              </span>
              <span style={{ color: colors.body, fontFamily: 'monospace' }}>{l.message}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Timeline */}
      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Timeline ({timeline.length})</div>
        <div style={{ maxHeight: 160, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2 }}>
          {timeline.length === 0 && <div style={{ fontSize: 12, color: colors.bodyMuted }}>No timeline events yet.</div>}
          {timeline.map((t) => (
            <div key={t.id} style={{ display: 'flex', gap: 8, fontSize: 11 }}>
              <span style={{ color: colors.inkMuted48, fontVariantNumeric: 'tabular-nums' }}>{t.ts}</span>
              <span style={{ color: colors.body }}>{t.event}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};