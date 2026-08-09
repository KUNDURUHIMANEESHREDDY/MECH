import React, { useState } from 'react';
import { Scale, Play, Loader2, Download } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useModel } from '../shared/hooks/useModel';
import { useWorkspaceStore } from '../shared/stores/workspace';

export interface EvidenceRow {
  ts: string;
  prompt: string;
  generated: string;
  tokens: number;
  gpuUtil?: number;
  memoryUtil?: number;
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

export const EvidenceFusionView: React.FC = () => {
  const { state: model, infer } = useModel();
  const { console: consoleLogs } = useWorkspaceStore();
  const [evidence, setEvidence] = useState<EvidenceRow[]>([]);
  const [running, setRunning] = useState(false);

  const PROBES = ['The capital of France is', 'When Mary and John went to the store, John gave the bag to', 'The first President of the United States was'];

  const fuse = async () => {
    if (!model.loaded || running) return;
    setRunning(true);
    const rows: EvidenceRow[] = [];
    for (const p of PROBES) {
      try {
        const r = await infer(p, 8);
        rows.push({ ts: new Date().toLocaleTimeString(), prompt: p, generated: r.generatedText, tokens: r.tokens.length, gpuUtil: r.gpuUtil, memoryUtil: r.memoryUtil });
      } catch (e) {
        rows.push({ ts: new Date().toLocaleTimeString(), prompt: p, generated: `ERR: ${(e as Error).message}`, tokens: 0 });
      }
    }
    setEvidence(rows);
    setRunning(false);
  };

  const download = () => {
    const payload = JSON.stringify(
      { gatheredAt: new Date().toISOString(), model: model.modelInfo?.model_name ?? null, evidence, consoleTail: consoleLogs.slice(-10) },
      null,
      2
    );
    const blob = new Blob([payload], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `evidence-fusion-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const errorCount = consoleLogs.filter((c) => c.level === 'error').length;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Scale size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Evidence Fusion</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{evidence.length} rows · {errorCount} console errors</span>
      </div>

      <div style={card}>
        <div style={{ fontWeight: 700, color: colors.ink }}>Corpus probes</div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button onClick={fuse} disabled={!model.loaded || running} style={{ background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: !model.loaded || running ? 0.5 : 1 }}>
            {running ? <Loader2 size={13} /> : <Play size={13} />} {running ? 'Running…' : `Run ${PROBES.length} probes`}
          </button>
          {evidence.length > 0 && (
            <button onClick={download} style={{ background: 'none', border: `1px solid ${colors.hairline}`, color: colors.bodyMuted, borderRadius: 6, padding: '6px 12px', fontSize: 12, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <Download size={13} /> Export JSON
            </button>
          )}
          {!model.loaded && <span style={{ fontSize: 11, color: colors.bodyMuted }}>Load a model first</span>}
        </div>
        <div style={{ fontSize: 11, color: colors.bodyMuted }}>
          Runs the corpus probe set through the loaded model and collects outputs with timing for the evidence ledger.
        </div>
      </div>

      {evidence.length === 0 ? (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          No evidence gathered yet. Run the probe batch to build the evidence ledger.
        </div>
      ) : (
        evidence.map((e) => (
          <div key={e.ts + e.prompt} style={card}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
              <span style={{ fontFamily: 'monospace', fontSize: 12, color: colors.ink }}>{e.prompt}</span>
              <span style={{ fontSize: 11, color: colors.bodyMuted, whiteSpace: 'nowrap' }}>{e.ts} · {e.tokens} tok</span>
            </div>
            <div style={{ fontFamily: 'Georgia, serif', fontSize: 14, color: colors.body, lineHeight: 1.5 }}>{e.generated}</div>
          </div>
        ))
      )}

      {model.modelInfo && (
        <div style={card}>
          <div style={{ fontWeight: 700, color: colors.ink }}>Evidence context</div>
          <div style={{ fontSize: 12, display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            <span style={{ color: colors.bodyMuted }}>Model <b style={{ color: colors.ink }}>{model.modelInfo.model_name}</b></span>
            <span style={{ color: colors.bodyMuted }}>Layers <b style={{ color: colors.ink }}>{model.modelInfo.num_layers ?? '–'}</b></span>
            <span style={{ color: colors.bodyMuted }}>Heads <b style={{ color: colors.ink }}>{model.modelInfo.num_heads ?? '–'}</b></span>
            <span style={{ color: colors.bodyMuted }}>Hidden <b style={{ color: colors.ink }}>{model.modelInfo.hidden_dim ?? '–'}</b></span>
          </div>
        </div>
      )}
    </div>
  );
};
