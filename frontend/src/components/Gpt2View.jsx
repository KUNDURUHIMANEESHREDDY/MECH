import React, { useState, useCallback, useRef } from 'react';
import { colors } from '../design/tokens/colors';

// ─── tiny helpers ──────────────────────────────────────────────────────────────

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgba(hex, a) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

function StatusBadge({ value }) {
  const ok = value === 'ok' || value === 'loaded';
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 4,
      fontSize: 11, fontWeight: 600, letterSpacing: '0.04em',
      background: ok ? colors.successSoft : colors.dangerSoft,
      color: ok ? colors.successText : colors.dangerText}}>
      {value || '—'}
    </span>
  );
}

function Spinner() {
  return (
    <span style={{ marginLeft: 6, opacity: 0.6, display: 'inline-flex', verticalAlign: 'middle' }}>
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 1h8M2 11h8M2 1l4 5 4-5M2 11l4-5 4 5" />
      </svg>
    </span>
  );
}

function Section({ title, children, badge }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 14, fontWeight: 700 }}>{title}</h3>
        {badge}
      </div>
      {children}
    </div>
  );
}

function KV({ k, v, mono }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', marginBottom: 3, fontSize: 13 }}>
      <span style={{ color: colors.inkMuted48, minWidth: 180, flexShrink: 0 }}>{k}</span>
      <span style={mono ? { fontFamily: 'monospace' } : {}}>{v ?? '—'}</span>
    </div>
  );
}

// ─── attention heatmap ─────────────────────────────────────────────────────────

function AttentionHeatmap({ matrix, tokens }) {
  if (!matrix || !matrix.length) return <p className="hint">No data yet.</p>;
  const max = Math.max(...matrix.flat());
  const cell = (v) => {
    // sqrt spreads low values so weak attention stays visible on dark theme
    const t = max > 0 ? Math.sqrt(Math.max(0, v) / max) : 0;
    const stops = [
      [30, 41, 59],
      [14, 165, 233],
      [250, 204, 21],
    ];
    const scaled = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(scaled));
    const f = scaled - i;
    const [r, g, b] = stops[i].map((ch, k) => Math.round(ch + (stops[i + 1][k] - ch) * f));
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    return { background: `rgb(${r},${g},${b})`, color: lum > 140 ? colors.ink : colors.onDark };
  };
  const sz = 36;
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
        <thead>
          <tr>
            <th style={{ width: sz }} />
            {tokens.map((t, i) => (
              <th key={i} style={{ width: sz, textAlign: 'center', fontWeight: 500, padding: '2px 1px',
                maxWidth: sz, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                title={t}>{t.slice(0, 5)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, qi) => (
            <tr key={qi}>
              <td style={{ fontSize: 11, fontWeight: 500, paddingRight: 4, whiteSpace: 'nowrap',
                maxWidth: 60, overflow: 'hidden', textOverflow: 'ellipsis' }} title={tokens[qi]}>
                {(tokens[qi] || '').slice(0, 6)}
              </td>
              {row.map((v, ki) => (
                <td key={ki} style={{
                  width: sz, height: sz, textAlign: 'center', ...cell(v),
                  border: '1px solid rgba(148,163,184,0.15)', borderRadius: 2
                }}>
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── logit bar ─────────────────────────────────────────────────────────────────

function LogitBar({ label, value, max, color }) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5, fontSize: 13 }}>
      <span style={{ minWidth: 70, textAlign: 'right' }}>{label}</span>
      <div style={{ flex: 1, background: colors.dividerSoft, borderRadius: 3, height: 14, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color || colors.primary, borderRadius: 3 }} />
      </div>
      <span style={{ minWidth: 50, fontFamily: 'monospace', fontSize: 12 }}>{value.toFixed(3)}</span>
    </div>
  );
}

// ─── main view ─────────────────────────────────────────────────────────────────

export default function Gpt2View({ api }) {
  // ── step 2+3: model ──
  const [modelInfo, setModelInfo]   = useState(null);
  const [loadingModel, setLoadingModel] = useState(false);

  // ── step 4: prompt ──
  const [prompt, setPrompt]   = useState('The capital of France is');
  const [promptResult, setPromptResult] = useState(null);
  const [runningPrompt, setRunningPrompt] = useState(false);

  // ── step 5: activations ──
  const [activLayer, setActivLayer] = useState(0);
  const [activResult, setActivResult] = useState(null);
  const [runningActiv, setRunningActiv] = useState(false);

  // ── step 6: attention ──
  const [attnLayer, setAttnLayer] = useState(10);
  const [attnHead,  setAttnHead]  = useState(7);
  const [attnResult, setAttnResult] = useState(null);
  const [runningAttn, setRunningAttn] = useState(false);

  // ── step 7: patching ──
  const [patchLayer,    setPatchLayer]    = useState(10);
  const [patchHead,     setPatchHead]     = useState(7);
  const [patchPosToken, setPatchPosToken] = useState(' Paris');
  const [patchNegToken, setPatchNegToken] = useState(' London');
  const [patchResult,   setPatchResult]   = useState(null);
  const [runningPatch,  setRunningPatch]  = useState(false);

  // ── step 8: IOI ──
  const [ioiIO,   setIoiIO]   = useState('Mary');
  const [ioiSubj, setIoiSubj] = useState('John');
  const [ioiResult, setIoiResult] = useState(null);
  const [runningIoi, setRunningIoi] = useState(false);

  const gpt2 = api;   // shorthand

  // ── step 2+3 handler ──
  const handleLoad = useCallback(async () => {
    setLoadingModel(true);
    try {
      const r = await gpt2.gpt2Load();
      setModelInfo(r);
    } catch (e) {
      setModelInfo({ status: 'error', error: e.message });
    } finally {
      setLoadingModel(false);
    }
  }, [gpt2]);

  // ── step 4 handler ──
  const handleRunPrompt = useCallback(async () => {
    if (!prompt.trim()) return;
    setRunningPrompt(true);
    setPromptResult(null);
    try {
      const r = await gpt2.gpt2RunPrompt(prompt);
      setPromptResult(r);
      // auto-refresh activations + attention with new cache
      setActivResult(null);
      setAttnResult(null);
      setPatchResult(null);
    } catch (e) {
      setPromptResult({ status: 'error', error: e.message });
    } finally {
      setRunningPrompt(false);
    }
  }, [gpt2, prompt]);

  // ── step 5 handler ──
  const handleGetActivations = useCallback(async () => {
    setRunningActiv(true);
    try {
      const r = await gpt2.gpt2GetActivations(activLayer);
      setActivResult(r);
    } catch (e) {
      setActivResult({ status: 'error', error: e.message });
    } finally {
      setRunningActiv(false);
    }
  }, [gpt2, activLayer]);

  // ── step 6 handler ──
  const handleGetAttn = useCallback(async () => {
    setRunningAttn(true);
    try {
      const r = await gpt2.gpt2AttentionHead(attnLayer, attnHead);
      setAttnResult(r);
    } catch (e) {
      setAttnResult({ status: 'error', error: e.message });
    } finally {
      setRunningAttn(false);
    }
  }, [gpt2, attnLayer, attnHead]);

  // ── step 7 handler ──
  const handlePatch = useCallback(async () => {
    setRunningPatch(true);
    try {
      const r = await gpt2.gpt2PatchHead(patchLayer, patchHead, patchPosToken, patchNegToken);
      setPatchResult(r);
    } catch (e) {
      setPatchResult({ status: 'error', error: e.message });
    } finally {
      setRunningPatch(false);
    }
  }, [gpt2, patchLayer, patchHead, patchPosToken, patchNegToken]);

  // ── step 8 handler ──
  const handleIoi = useCallback(async () => {
    setRunningIoi(true);
    try {
      const r = await gpt2.gpt2RunIoi(ioiIO, ioiSubj);
      setIoiResult(r);
    } catch (e) {
      setIoiResult({ status: 'error', error: e.message });
    } finally {
      setRunningIoi(false);
    }
  }, [gpt2, ioiIO, ioiSubj]);

  const loaded = modelInfo?.status === 'loaded';
  const hasCache = promptResult?.status === 'ok';

  const top16 = promptResult?.top16 || [];
  const maxLogit = top16.length ? Math.max(...top16.map(t => Math.abs(t.logit)), 0.01) : 0.01;

  return (
    <div style={{ padding: '20px 24px', maxWidth: 900 }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>GPT-2 Small — Live Interpretability</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: colors.inkMuted48 }}>
          8-step mechanistic interpretability pipeline running on the real model via Python sidecar
        </p>
      </div>

      {/* ── STEP 2 + 3: Load Model ─────────────────────────────────────────── */}
      <Section
        title="Steps 2 & 3 — Load GPT-2 Small"
        badge={modelInfo && <StatusBadge value={modelInfo.status} />}
      >
        <button className="btn btn-primary" onClick={handleLoad} disabled={loadingModel}>
          {loadingModel ? <>Loading…<Spinner /></> : loaded ? 'Reload Model' : 'Load GPT-2 Small'}
        </button>

        {modelInfo && modelInfo.status === 'loaded' && (
          <div style={{ marginTop: 12 }}>
            <KV k="Model" v={modelInfo.model_name} />
            <KV k="Layers"          v={modelInfo.n_layers} />
            <KV k="Attention heads" v={modelInfo.n_heads} />
            <KV k="Embedding dim"   v={modelInfo.d_model} />
            <KV k="MLP dim"         v={modelInfo.d_mlp} />
            <KV k="Device"          v={modelInfo.device} />
          </div>
        )}
        {modelInfo?.status === 'error' && (
          <p style={{ color: colors.danger, marginTop: 8, fontSize: 12 }}>{modelInfo.error}</p>
        )}
      </Section>

      {/* ── STEP 4: Run Prompt ────────────────────────────────────────────── */}
      <Section
        title="Step 4 — Run Prompt"
        badge={promptResult && <StatusBadge value={promptResult.status} />}
      >
        <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
          <input
            className="input-text"
            style={{ flex: 1 }}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleRunPrompt()}
            placeholder="Enter a prompt…"
          />
          <button className="btn btn-primary" onClick={handleRunPrompt} disabled={runningPrompt || !loaded}>
            {runningPrompt ? <>Running…<Spinner /></> : 'Run'}
          </button>
        </div>

        {promptResult?.status === 'ok' && (
          <>
             <KV k="Tokens" v={promptResult.str_tokens?.join(' | ')} mono />
             <KV k="Top-1 token" v={`"${promptResult.top5?.[0]?.token}"`} />
             <KV k="Next token"  v={`"${promptResult.next_token}"`} />
             <div style={{ marginTop: 12 }}>
               <p style={{ fontSize: 12, fontWeight: 600, marginBottom: 6, color: colors.inkMuted48 }}>
                 Top-16 predictions
               </p>
               {top16.map((t, i) => (
                 <LogitBar key={t.token} label={t.token} value={t.logit} max={maxLogit}
                   color={i === 0 ? colors.success : colors.inkMuted48} />
               ))}
             </div>
            <div style={{ marginTop: 12 }}>
              <p style={{ fontSize: 12, fontWeight: 600, marginBottom: 4, color: colors.inkMuted48 }}>
                Top-5 predictions
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {promptResult.top5?.map((t, i) => (
                  <span key={i} style={{
                    padding: '3px 8px', borderRadius: 4, fontSize: 12,
                    background: i === 0 ? colors.infoSoft : colors.dividerSoft,
                    fontFamily: 'monospace', fontWeight: i === 0 ? 700 : 400
                  }}>
                    "{t.token}" {t.logit.toFixed(2)}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}
        {promptResult?.status === 'error' && (
          <p style={{ color: colors.danger, fontSize: 12 }}>{promptResult.error}</p>
        )}
      </Section>

      {/* ── STEP 5: Activations ───────────────────────────────────────────── */}
      <Section
        title="Step 5 — Capture Activations"
        badge={activResult && <StatusBadge value={activResult.status} />}
      >
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
          <label style={{ fontSize: 13 }}>Layer</label>
          <input type="number" className="input-text" style={{ width: 70 }}
            value={activLayer} min={0} max={11}
            onChange={(e) => setActivLayer(Number(e.target.value))} />
          <button className="btn btn-secondary" onClick={handleGetActivations}
            disabled={runningActiv || !hasCache}>
            {runningActiv ? <>Fetching…<Spinner /></> : 'Fetch shapes'}
          </button>
        </div>
        {!hasCache && <p className="hint">Run a prompt first to populate the activation cache.</p>}
        {activResult?.status === 'ok' && (
          <>
            <KV k={`resid_post[L${activResult.layer}] shape`} v={JSON.stringify(activResult.resid_shape)} mono />
            <KV k={`attn_pattern[L${activResult.layer}] shape`} v={JSON.stringify(activResult.attn_shape)} mono />
            <KV k={`mlp_post[L${activResult.layer}] shape`}    v={JSON.stringify(activResult.mlp_shape)} mono />
          </>
        )}
        {activResult?.status === 'error' && (
          <p style={{ color: colors.danger, fontSize: 12 }}>{activResult.error}</p>
        )}
      </Section>

      {/* ── STEP 6: Attention ─────────────────────────────────────────────── */}
      <Section
        title="Step 6 — Attention Head Visualisation"
        badge={attnResult && <StatusBadge value={attnResult.status} />}
      >
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
          <label style={{ fontSize: 13 }}>Layer</label>
          <input type="number" className="input-text" style={{ width: 60 }}
            value={attnLayer} min={0} max={11}
            onChange={(e) => setAttnLayer(Number(e.target.value))} />
          <label style={{ fontSize: 13 }}>Head</label>
          <input type="number" className="input-text" style={{ width: 60 }}
            value={attnHead} min={0} max={11}
            onChange={(e) => setAttnHead(Number(e.target.value))} />
          <button className="btn btn-secondary" onClick={handleGetAttn}
            disabled={runningAttn || !hasCache}>
            {runningAttn ? <>Loading…<Spinner /></> : 'Show heatmap'}
          </button>
        </div>
        {!hasCache && <p className="hint">Run a prompt first.</p>}
        {attnResult?.status === 'ok' && (
          <>
            <p style={{ fontSize: 12, color: colors.inkMuted48, marginBottom: 8 }}>
              Layer {attnResult.layer} · Head {attnResult.head} — rows = query, cols = key
            </p>
            <AttentionHeatmap matrix={attnResult.matrix} tokens={attnResult.str_tokens || []} />
          </>
        )}
        {attnResult?.status === 'error' && (
          <p style={{ color: colors.danger, fontSize: 12 }}>{attnResult.error}</p>
        )}
      </Section>

      {/* ── STEP 7: Patching ──────────────────────────────────────────────── */}
      <Section
        title="Step 7 — Activation Patching"
        badge={patchResult && <StatusBadge value={patchResult.status} />}
      >
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8, marginBottom: 10 }}>
          <div>
            <label style={{ fontSize: 12 }}>Layer</label>
            <input type="number" className="input-text" style={{ width: '100%' }}
              value={patchLayer} min={0} max={11}
              onChange={(e) => setPatchLayer(Number(e.target.value))} />
          </div>
          <div>
            <label style={{ fontSize: 12 }}>Head</label>
            <input type="number" className="input-text" style={{ width: '100%' }}
              value={patchHead} min={0} max={11}
              onChange={(e) => setPatchHead(Number(e.target.value))} />
          </div>
          <div>
            <label style={{ fontSize: 12 }}>Positive token</label>
            <input type="text" className="input-text" style={{ width: '100%' }}
              value={patchPosToken}
              onChange={(e) => setPatchPosToken(e.target.value)} />
          </div>
          <div>
            <label style={{ fontSize: 12 }}>Negative token</label>
            <input type="text" className="input-text" style={{ width: '100%' }}
              value={patchNegToken}
              onChange={(e) => setPatchNegToken(e.target.value)} />
          </div>
        </div>
        <button className="btn btn-primary" onClick={handlePatch}
          disabled={runningPatch || !hasCache}>
          {runningPatch ? <>Patching…<Spinner /></> : 'Zero-ablate head'}
        </button>
        {!hasCache && <span style={{ fontSize: 12, color: colors.inkMuted48, marginLeft: 10 }}>
          Run a prompt first.
        </span>}

        {patchResult?.status === 'ok' && (
          <div style={{ marginTop: 12 }}>
            <KV k="Clean logit diff"   v={patchResult.clean_ld?.toFixed(4)} mono />
            <KV k="Patched logit diff" v={patchResult.patched_ld?.toFixed(4)} mono />
            <KV k="Delta"              v={patchResult.delta?.toFixed(4)} mono />
            <div style={{
              marginTop: 10, padding: '8px 12px', borderRadius: 6, fontSize: 13,
background: patchResult.direction === 'hurts' ? colors.dangerSoft : colors.successSoft,
      color:      patchResult.direction === 'hurts' ? colors.dangerText : colors.successText,
              fontWeight: 600}}>
              Zeroing L{patchResult.layer}H{patchResult.head}{' '}
              <strong>{patchResult.direction}</strong> the{' '}
              "{(patchPosToken||'').trim()}" prediction by{' '}
              {Math.abs(patchResult.delta || 0).toFixed(4)} logits
            </div>
          </div>
        )}
        {patchResult?.status === 'error' && (
          <p style={{ color: colors.danger, fontSize: 12, marginTop: 8 }}>{patchResult.error}</p>
        )}
      </Section>

      {/* ── STEP 8: IOI ───────────────────────────────────────────────────── */}
      <Section
        title="Step 8 — IOI Experiment"
        badge={ioiResult && <StatusBadge value={ioiResult.status} />}
      >
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
          <label style={{ fontSize: 13 }}>IO name</label>
          <input type="text" className="input-text" style={{ width: 90 }}
            value={ioiIO} onChange={(e) => setIoiIO(e.target.value)} />
          <label style={{ fontSize: 13 }}>Subject name</label>
          <input type="text" className="input-text" style={{ width: 90 }}
            value={ioiSubj} onChange={(e) => setIoiSubj(e.target.value)} />
          <button className="btn btn-primary" onClick={handleIoi}
            disabled={runningIoi || !loaded}>
            {runningIoi ? <>Running…<Spinner /></> : 'Run IOI'}
          </button>
        </div>

        {ioiResult?.status === 'ok' && (
          <div>
            <div style={{ marginBottom: 12 }}>
              <p style={{ fontSize: 12, color: colors.inkMuted48, marginBottom: 4 }}>
                Clean: <code style={{ background: colors.dividerSoft, padding: '1px 4px', borderRadius: 3 }}>
                  {ioiResult.clean_prompt}
                </code>
              </p>
              <KV k="Top-1 prediction"      v={`"${ioiResult.clean_top1}"`} />
              <KV k={`Logit diff (${ioiResult.io_name} − ${ioiResult.subj_name})`}
                v={ioiResult.clean_ld?.toFixed(4)} mono />
              <div style={{
                display: 'inline-block', padding: '3px 10px', borderRadius: 4, fontSize: 12,
                marginTop: 4, fontWeight: 600,
                background: ioiResult.ioi_pass ? colors.successSoft : colors.dangerSoft,
                color:      ioiResult.ioi_pass ? colors.successText : colors.dangerText}}>
                {ioiResult.ioi_pass ? `Model predicts ${ioiResult.io_name} (IO) — PASS` : 'FAIL'}
              </div>
            </div>
            <div>
              <p style={{ fontSize: 12, color: colors.inkMuted48, marginBottom: 4 }}>
                Corrupted: <code style={{ background: colors.dividerSoft, padding: '1px 4px', borderRadius: 3 }}>
                  {ioiResult.corrupted_prompt}
                </code>
              </p>
              <KV k="Top-1 prediction"      v={`"${ioiResult.corrupted_top1}"`} />
              <KV k={`Logit diff (${ioiResult.io_name} − ${ioiResult.subj_name})`}
                v={ioiResult.corrupted_ld?.toFixed(4)} mono />
              <div style={{
                display: 'inline-block', padding: '3px 10px', borderRadius: 4, fontSize: 12,
                marginTop: 4, fontWeight: 600,
                background: ioiResult.corrupted_pass ? colors.successSoft : colors.dangerSoft,
                color:      ioiResult.corrupted_pass ? colors.successText : colors.dangerText}}>
                {ioiResult.corrupted_pass
                  ? `Model flips to ${ioiResult.subj_name} (Subject) — PASS`
                  : 'FAIL'}
              </div>
            </div>
          </div>
        )}
        {ioiResult?.status === 'error' && (
          <p style={{ color: colors.danger, fontSize: 12 }}>{ioiResult.error}</p>
        )}
      </Section>
    </div>
  );
}
