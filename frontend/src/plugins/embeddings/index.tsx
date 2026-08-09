import React, { useMemo, useState } from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { useModel } from '../../shared/hooks/useModel';
import { useSelectionStore } from '../../shared/stores/selection';
import { colors } from '../../design/tokens/colors';

/**
 * Builds a real contextual vector per token from the loaded model result:
 * for a chosen layer, every token's vector is the row of the attention
 * matrix averaged across all heads (the "how this token reads the
 * sequence" signature). Cosine similarity between these vectors drives
 * the similarity matrix.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? dot / denom : 0;
}

const EmbeddingViewerBody: FC<PanelContext> = () => {
  const { state: model } = useModel();
  const sel = useSelectionStore();
  const [layerIdx, setLayerIdx] = useState<number | null>(null);

  const tokens = model.result?.tokens.map(t => t.text) ?? [];
  const layers = model.result?.layers ?? [];

  const selectedLayer = layerIdx ?? sel.layer ?? (layers.length > 0 ? 0 : null);
  const layer = selectedLayer !== null ? layers[selectedLayer] : undefined;
  const seqN = Math.min(tokens.length, layer?.heads?.[0]?.attentionMatrix?.length ?? tokens.length);
  const seqTokens = tokens.slice(0, seqN);

  const { vectors, matrix, pairs } = useMemo(() => {
    if (!layer || seqN === 0) {
      return { vectors: [] as number[][], matrix: [] as number[][], pairs: [] as Array<{ i: number; j: number; score: number }> };
    }
    const n = seqN;
    const vectors: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
    for (const head of layer.heads) {
      const m = head.attentionMatrix;
      if (!m || m.length !== n) continue;
      for (let t = 0; t < n; t++) {
        for (let s = 0; s < n; s++) {
          vectors[t][s] += m[t][s] ?? 0;
        }
      }
    }
    // Normalise rows to unit vectors so cosine equals dot product.
    for (let t = 0; t < n; t++) {
      let mag = 0;
      for (let s = 0; s < n; s++) mag += vectors[t][s] * vectors[t][s];
      mag = Math.sqrt(mag) || 1;
      for (let s = 0; s < n; s++) vectors[t][s] /= mag;
    }
    const matrix = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => cosineSimilarity(vectors[i], vectors[j])),
    );
    const pairs: Array<{ i: number; j: number; score: number }> = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        pairs.push({ i, j, score: matrix[i][j] });
      }
    }
    pairs.sort((a, b) => b.score - a.score);
    return { vectors, matrix, pairs };
  }, [layer, tokens]);

  const cellSize = 22;
  const gridSize = tokens.length * cellSize;

  const maxAbs = useMemo(() => {
    let m = 0;
    for (const row of matrix) for (const v of row) m = Math.max(m, Math.abs(v));
    return m || 1;
  }, [matrix]);

  return (
    <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 12, overflow: 'auto', height: '100%', backgroundColor: colors.canvas }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4 }}>
          Token embeddings — cosine similarity
        </div>
        {layers.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: colors.bodyMuted }}>
            <span>Layer</span>
            <select
              value={selectedLayer ?? 0}
              onChange={e => setLayerIdx(parseInt(e.target.value, 10))}
              style={{
                padding: '5px 8px', borderRadius: 6, border: `1px solid ${colors.border}`,
                backgroundColor: colors.canvasParchment, color: colors.ink, fontSize: 12, outline: 'none',
              }}
            >
              {layers.map(l => <option key={l.index} value={l.index}>L{l.index}</option>)}
            </select>
          </div>
        )}
      </div>

      {seqN === 0 || matrix.length === 0 ? (
        <div style={{ fontSize: 13, color: colors.bodyMuted, lineHeight: 1.5 }}>
          No activations loaded yet. Run an inference to build real contextual
          token vectors from the attention maps and inspect their pairwise
          cosine similarity.
        </div>
      ) : (
        <>
          {/* Similarity matrix */}
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 8, padding: 12, backgroundColor: colors.surfaceTile1, overflow: 'auto' }}>
            <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.3 }}>
              {tokens.length} × {tokens.length} contextual similarity
            </div>
            <div style={{ width: gridSize, position: 'relative' }}>
              {matrix.map((row, i) => (
                <div key={`row-${i}`} style={{ display: 'flex' }}>
                  {row.map((v, j) => {
                    const t = (v + 1) / 2; // map [-1,1] -> [0,1]
                    const a = Math.max(0, Math.min(1, t));
                    return (
                      <div
                        key={`cell-${i}-${j}`}
                        title={`${tokens[i]} ~ ${tokens[j]} = ${v.toFixed(3)}`}
                        style={{
                          width: cellSize, height: cellSize, flex: '0 0 auto',
                          backgroundColor: v > 0
                            ? `rgba(37, 99, 235, ${a * 0.85 + 0.1})`
                            : `rgba(225, 29, 72, ${(1 - a) * 0.6})`,
                        }}
                      />
                    );
                  })}
                </div>
              ))}
              {/* Token labels along the top */}
              <div style={{ display: 'flex', position: 'absolute', top: -cellSize - 4, left: 0 }}>
                {seqTokens.map((t, j) => (
                  <div key={`tl-${j}`} style={{ width: cellSize, fontSize: 9, color: colors.bodyMuted, whiteSpace: 'nowrap', overflow: 'hidden' }}>
                    {t}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Most similar pairs */}
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 8, padding: 12, backgroundColor: colors.surfaceTile1 }}>
            <div style={{ fontSize: 11, color: colors.bodyMuted, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.3 }}>
              Most similar token pairs
            </div>
            {pairs.slice(0, 10).map((p, idx) => (
              <div key={`pair-${idx}`} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span style={{ width: 22, fontSize: 11, color: colors.bodyMuted, textAlign: 'right' }}>{idx + 1}</span>
                <span style={{ width: 160, fontSize: 12, color: colors.ink, fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {seqTokens[p.i]} ↔ {seqTokens[p.j]}
                </span>
                <div style={{ flex: 1, height: 8, backgroundColor: colors.surfacePearl, borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${(p.score / maxAbs) * 100}%`, backgroundColor: colors.primary, borderRadius: 4 }} />
                </div>
                <span style={{ width: 44, fontSize: 11, color: colors.bodyMuted, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                  {p.score.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

pluginRegistry.register({
  id: 'embedding_viewer',
  title: 'Embedding Viewer',
  icon: 'Globe',
  category: 'embeddings',
  resourceKinds: ['model', 'token'],
  defaultDock: 'center',
  Body: EmbeddingViewerBody,
});