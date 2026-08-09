import React, { useMemo, useState } from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { useModel } from '../../shared/hooks/useModel';
import { api } from '../../services/api';
import { colors } from '../../design/tokens/colors';

interface BackendToken {
  token?: string;
  text?: string;
  logit?: number;
  prob?: number;
}

const LogitLensBody: FC<PanelContext> = () => {
  const { state: model } = useModel();
  const [prompt, setPrompt] = useState('');
  const [topTokens, setTopTokens] = useState<Array<{ text: string; score: number }>>([]);
  const [busy, setBusy] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);

  const tokens = model.result?.tokens.map(t => t.text) ?? [];
  const layers = model.result?.layers ?? [];

  // Per-layer sweep computed from real neuron token activations: for every
  // layer we measure how much signal each token position carries, then rank
  // the tokens as the model's "focus" progressing towards the final logits.
  const layerSweep = useMemo(() => {
    return layers.map((layer) => {
      const neurons = layer.heads.flatMap(h => h.neurons);
      if (neurons.length === 0 || tokens.length === 0) {
        return { layer: layer.index, entries: [] as Array<{ token: string; score: number }> };
      }
      const energy = tokens.map(() => 0);
      for (const n of neurons) {
        if (!n.tokenActivations) continue;
        n.tokenActivations.forEach((a, i) => {
          if (i < energy.length) energy[i] += a;
        });
      }
      const max = Math.max(...energy, 1e-9);
      const entries = energy
        .map((score, i) => ({ token: tokens[i], score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 3)
        .map(e => ({ ...e, score: e.score / max }));
      return { layer: layer.index, entries };
    });
  }, [layers, tokens]);

  const runBackend = async () => {
    if (!prompt.trim()) return;
    setBusy(true);
    setBackendError(null);
    try {
      const raw = (await api.gpt2RunPrompt(prompt)) as Record<string, unknown>;
      const candidates: BackendToken[] = Array.isArray(raw.top16)
        ? (raw.top16 as BackendToken[])
        : Array.isArray(raw.top5)
          ? (raw.top5 as BackendToken[])
          : Array.isArray(raw.top_tokens)
            ? (raw.top_tokens as BackendToken[])
            : [];
      const rows = candidates
        .map(c => ({
          text: c.token ?? c.text ?? '?',
          score: typeof c.logit === 'number' ? c.logit : (typeof c.prob === 'number' ? c.prob : 0),
        }))
        .filter(r => r.text !== '?');
      setTopTokens(rows);
      if (rows.length === 0) setBackendError('Backend returned no top-token logits.');
    } catch (e: any) {
      setBackendError(e.message || 'Backend unavailable.');
    } finally {
      setBusy(false);
    }
  };

  const maxLogit = Math.max(...topTokens.map(t => Math.abs(t.score)), 1e-9);

  return (
    <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 14, overflow: 'auto', height: '100%', backgroundColor: colors.canvas }}>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') runBackend(); }}
          placeholder="Run a prompt to read top logits…"
          style={{
            flex: 1, padding: '7px 10px', border: `1px solid ${colors.border}`,
            borderRadius: 6, backgroundColor: colors.canvasParchment, color: colors.ink,
            fontSize: 12, outline: 'none',
          }}
        />
        <button
          onClick={runBackend}
          disabled={busy}
          style={{
            padding: '7px 14px', borderRadius: 6, border: 'none',
            backgroundColor: busy ? colors.bodyMuted : colors.primary, color: colors.onPrimary,
            fontSize: 12, fontWeight: 600, cursor: busy ? 'default' : 'pointer',
          }}
        >
          {busy ? 'Reading…' : 'Read Logits'}
        </button>
      </div>

      {backendError && (
        <div style={{ padding: '8px 10px', borderRadius: 6, backgroundColor: colors.dangerSoft, border: `1px solid ${colors.dangerBorder}`, color: colors.dangerText, fontSize: 12 }}>
          {backendError}
        </div>
      )}

      {topTokens.length > 0 && (
        <div style={{ border: `1px solid ${colors.border}`, borderRadius: 8, backgroundColor: colors.surfaceTile1, padding: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 0.4 }}>
            Final-layer logit lens
          </div>
          {topTokens.map((t, i) => (
            <div key={`${t.text}-${i}`} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
              <span style={{ width: 26, fontSize: 11, color: colors.bodyMuted, textAlign: 'right' }}>{i + 1}</span>
              <span style={{ width: 120, fontSize: 12, color: colors.ink, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.text}</span>
              <div style={{ flex: 1, height: 10, backgroundColor: colors.surfacePearl, borderRadius: 5, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${Math.min(100, (Math.abs(t.score) / maxLogit) * 100)}%`, backgroundColor: colors.primary, borderRadius: 5 }} />
              </div>
              <span style={{ width: 64, fontSize: 11, color: colors.bodyMuted, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{t.score.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 8, backgroundColor: colors.surfaceTile1, padding: 12 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 0.4 }}>
          Layer sweep {layers.length > 0 ? `— ${layers.length} layers` : ''}
        </div>
        {layerSweep.length === 0 ? (
          <div style={{ fontSize: 13, color: colors.bodyMuted }}>
            Run an inference to populate layer activations. Each row shows the tokens each layer focuses on most on the way to the prediction.
          </div>
        ) : (
          layerSweep.map((row) => (
            <div key={row.layer} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ width: 22, fontSize: 11, color: colors.bodyMuted, textAlign: 'right' }}>{row.layer}</span>
              <div style={{ flex: 1, display: 'flex', gap: 4 }}>
                {row.entries.length === 0 ? (
                  <span style={{ fontSize: 11, color: colors.bodyMuted }}>—</span>
                ) : (
                  row.entries.map((e2, i) => (
                    <div key={i} style={{
                      flex: e2.score, height: 18, borderRadius: 4,
                      backgroundColor: i === 0 ? colors.purple : colors.purpleSoft,
                      display: 'flex', alignItems: 'center', paddingLeft: 6, overflow: 'hidden',
                      fontSize: 10, color: i === 0 ? colors.onPrimary : colors.purpleText, whiteSpace: 'nowrap',
                    }}>
                      {e2.token}
                    </div>
                  ))
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

pluginRegistry.register({
  id: 'logit_lens',
  title: 'Logit Lens',
  icon: 'Search',
  category: 'logit-lens',
  resourceKinds: ['model', 'token'],
  defaultDock: 'bottom',
  Body: LogitLensBody,
});