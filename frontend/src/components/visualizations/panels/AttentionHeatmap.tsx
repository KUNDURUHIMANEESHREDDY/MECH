import React, { useMemo, useState, useRef, useEffect, useCallback } from 'react';
import { colors } from '../../../design/tokens/colors';
import { Eye, Grid, Layers, Sliders, AlertCircle } from 'lucide-react';

export interface AttentionHeatmapProps {
  matrix?: number[][];
  tokens?: string[];
  hoveredToken?: number | null;
  onHoverToken?: (i: number | null) => void;
  allLayers?: Array<{
    index: number;
    heads: Array<{
      index: number;
      attentionMatrix: number[][];
    }>;
  }>;
  selectedLayer?: number;
  selectedHead?: number;
  onSelectLayerHead?: (layer: number, head: number) => void;
}

// Canonical BertViz Head Color Palette (Jesse Vig / D3 10-Categorical)
const BERTVIZ_HEAD_COLORS = [
  '#2563eb', // Head 0: Blue
  '#ea580c', // Head 1: Orange
  '#16a34a', // Head 2: Green
  '#dc2626', // Head 3: Red
  '#9333ea', // Head 4: Purple
  '#854d0e', // Head 5: Brown
  '#db2777', // Head 6: Pink
  '#4b5563', // Head 7: Gray
  '#ca8a04', // Head 8: Yellow/Olive
  '#0891b2', // Head 9: Cyan
  '#4f46e5', // Head 10: Indigo
  '#059669', // Head 11: Teal
];

type BertVizViewMode = 'head' | 'model' | 'heatmap';

export function AttentionHeatmap({
  matrix = [],
  tokens = [],
  hoveredToken: externalHoveredToken = null,
  onHoverToken,
  allLayers = [],
  selectedLayer: propLayer = 0,
  selectedHead: propHead = 0,
  onSelectLayerHead,
}: AttentionHeatmapProps) {
  const [viewMode, setViewMode] = useState<BertVizViewMode>('head');
  const [internalHoveredToken, setInternalHoveredToken] = useState<number | null>(null);
  const [activeQueryToken, setActiveQueryToken] = useState<number | null>(null);
  const [activeKeyToken, setActiveKeyToken] = useState<number | null>(null);
  const [attentionThreshold, setAttentionThreshold] = useState<number>(0.02);
  
  // Active layer and head state
  const [currentLayer, setCurrentLayer] = useState<number>(propLayer);
  const [currentHead, setCurrentHead] = useState<number>(propHead);

  // Multi-head toggles for BertViz Head View (Set of active head indices in current layer)
  const [enabledHeads, setEnabledHeads] = useState<Set<number>>(() => new Set([propHead % 12]));

  const hoveredToken = externalHoveredToken !== null ? externalHoveredToken : internalHoveredToken;

  const handleTokenHover = useCallback((idx: number | null) => {
    setInternalHoveredToken(idx);
    if (onHoverToken) {
      onHoverToken(idx);
    }
  }, [onHoverToken]);

  // Sync prop changes
  useEffect(() => {
    setCurrentLayer(propLayer);
    setCurrentHead(propHead);
    setEnabledHeads(new Set([propHead % 12]));
  }, [propLayer, propHead]);

  // Resolve current active attention matrix
  const resolvedMatrix = useMemo(() => {
    if (allLayers && allLayers.length > currentLayer && allLayers[currentLayer]?.heads?.length > currentHead) {
      const m = allLayers[currentLayer].heads[currentHead]?.attentionMatrix;
      if (m && m.length > 0) return m;
    }
    return matrix;
  }, [allLayers, currentLayer, currentHead, matrix]);

  // Fail-closed check: No valid attention weights available
  const hasLiveAttention = useMemo(() => {
    return resolvedMatrix && resolvedMatrix.length > 0 && tokens && tokens.length > 0;
  }, [resolvedMatrix, tokens]);

  const seqLen = tokens.length;

  const toggleHead = (headIdx: number) => {
    setEnabledHeads(prev => {
      const next = new Set(prev);
      if (next.has(headIdx)) {
        if (next.size > 1) {
          next.delete(headIdx);
        }
      } else {
        next.add(headIdx);
      }
      return next;
    });
  };

  const handleSelectHead = (layerIdx: number, headIdx: number) => {
    setCurrentLayer(layerIdx);
    setCurrentHead(headIdx);
    setEnabledHeads(new Set([headIdx]));
    if (onSelectLayerHead) {
      onSelectLayerHead(layerIdx, headIdx);
    }
  };

  // ---------------------------------------------------------------------------
  // Fail-Closed Abstention View
  // ---------------------------------------------------------------------------
  if (!hasLiveAttention) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 32,
          background: colors.canvas,
          borderRadius: 8,
          border: `1px dashed ${colors.hairline}`,
          color: colors.inkMuted48,
          gap: 12,
          minHeight: 240,
        }}
      >
        <AlertCircle size={28} color={colors.warning} />
        <div style={{ fontWeight: 600, color: colors.ink }}>
          EXECUTION REQUIRED: No Live Attention Tensors
        </div>
        <div style={{ fontSize: 12, maxWidth: 360, textAlign: 'center', lineHeight: 1.5 }}>
          Attention visualization requires live forward pass execution data from PyTorch.
          Synthetic or default attention patterns are prohibited.
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Main BertViz UI Shell
  // ---------------------------------------------------------------------------
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, width: '100%' }}>
      {/* Top Controls Toolbar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 8,
          padding: '8px 12px',
          background: colors.canvas,
          borderRadius: 6,
          border: `1px solid ${colors.hairline}`,
        }}
      >
        {/* Mode Selector */}
        <div style={{ display: 'flex', gap: 4 }}>
          <button
            onClick={() => setViewMode('head')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              border: `1px solid ${viewMode === 'head' ? colors.primary : colors.hairline}`,
              background: viewMode === 'head' ? colors.primary : colors.surfacePearl,
              color: viewMode === 'head' ? colors.onPrimary : colors.ink,
            }}
          >
            <Eye size={12} />
            BertViz Head View
          </button>
          <button
            onClick={() => setViewMode('model')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              border: `1px solid ${viewMode === 'model' ? colors.primary : colors.hairline}`,
              background: viewMode === 'model' ? colors.primary : colors.surfacePearl,
              color: viewMode === 'model' ? colors.onPrimary : colors.ink,
            }}
          >
            <Grid size={12} />
            Model View (All Heads)
          </button>
          <button
            onClick={() => setViewMode('heatmap')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '4px 10px',
              borderRadius: 4,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
              border: `1px solid ${viewMode === 'heatmap' ? colors.primary : colors.hairline}`,
              background: viewMode === 'heatmap' ? colors.primary : colors.surfacePearl,
              color: viewMode === 'heatmap' ? colors.onPrimary : colors.ink,
            }}
          >
            <Layers size={12} />
            Heatmap Matrix
          </button>
        </div>

        {/* Threshold Slider (for Head View) */}
        {viewMode === 'head' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: colors.ink }}>
            <Sliders size={12} />
            <span>Threshold: {(attentionThreshold * 100).toFixed(0)}%</span>
            <input
              type="range"
              min="0.0"
              max="0.30"
              step="0.01"
              value={attentionThreshold}
              onChange={e => setAttentionThreshold(parseFloat(e.target.value))}
              style={{ width: 80, accentColor: colors.primary }}
            />
          </div>
        )}
      </div>

      {/* Multi-Head Color Toggles (BertViz Head View) */}
      {viewMode === 'head' && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: colors.inkMuted48, marginRight: 4 }}>
            Layer {currentLayer} Heads:
          </span>
          {Array.from({ length: 12 }, (_, h) => {
            const isEnabled = enabledHeads.has(h);
            const headColor = BERTVIZ_HEAD_COLORS[h % BERTVIZ_HEAD_COLORS.length];
            return (
              <button
                key={h}
                onClick={() => toggleHead(h)}
                style={{
                  padding: '2px 8px',
                  borderRadius: 12,
                  fontSize: 10,
                  fontWeight: 700,
                  cursor: 'pointer',
                  border: `1px solid ${isEnabled ? headColor : colors.hairline}`,
                  background: isEnabled ? headColor : colors.canvas,
                  color: isEnabled ? '#ffffff' : colors.inkMuted48,
                  transition: 'all 0.15s ease',
                }}
              >
                H{h}
              </button>
            );
          })}
        </div>
      )}

      {/* View Mode Content */}
      {viewMode === 'head' && (
        <BertVizHeadView
          tokens={tokens}
          currentLayer={currentLayer}
          enabledHeads={enabledHeads}
          allLayers={allLayers}
          fallbackMatrix={resolvedMatrix}
          activeQueryToken={activeQueryToken}
          activeKeyToken={activeKeyToken}
          hoveredToken={hoveredToken}
          attentionThreshold={attentionThreshold}
          onHoverToken={handleTokenHover}
          onSelectQueryToken={idx => setActiveQueryToken(activeQueryToken === idx ? null : idx)}
          onSelectKeyToken={idx => setActiveKeyToken(activeKeyToken === idx ? null : idx)}
        />
      )}

      {viewMode === 'model' && (
        <BertVizModelView
          tokens={tokens}
          allLayers={allLayers}
          selectedLayer={currentLayer}
          selectedHead={currentHead}
          onSelectHead={handleSelectHead}
        />
      )}

      {viewMode === 'heatmap' && (
        <BertVizHeatmapMatrix
          matrix={resolvedMatrix}
          tokens={tokens}
          hoveredToken={hoveredToken}
          onHoverToken={handleTokenHover}
        />
      )}
    </div>
  );
}

// ============================================================================
// 1. BertViz Head View (Bipartite Attention Arc Graph)
// ============================================================================

interface BertVizHeadViewProps {
  tokens: string[];
  currentLayer: number;
  enabledHeads: Set<number>;
  allLayers: Array<{ index: number; heads: Array<{ index: number; attentionMatrix: number[][] }> }>;
  fallbackMatrix: number[][];
  activeQueryToken: number | null;
  activeKeyToken: number | null;
  hoveredToken: number | null;
  attentionThreshold: number;
  onHoverToken: (i: number | null) => void;
  onSelectQueryToken: (i: number | null) => void;
  onSelectKeyToken: (i: number | null) => void;
}

function BertVizHeadView({
  tokens,
  currentLayer,
  enabledHeads,
  allLayers,
  fallbackMatrix,
  activeQueryToken,
  activeKeyToken,
  hoveredToken,
  attentionThreshold,
  onHoverToken,
  onSelectQueryToken,
  onSelectKeyToken,
}: BertVizHeadViewProps) {
  const rowHeight = 24;
  const tokenWidth = 100;
  const svgWidth = 460;
  const svgHeight = Math.max(180, tokens.length * rowHeight + 20);

  const leftX = tokenWidth;
  const rightX = svgWidth - tokenWidth;

  // Gather active head matrices
  const activeHeadMatrices = useMemo(() => {
    const layerObj = allLayers && allLayers.length > currentLayer ? allLayers[currentLayer] : null;
    const result: Array<{ headIdx: number; matrix: number[][]; color: string }> = [];

    enabledHeads.forEach(headIdx => {
      let mat = fallbackMatrix;
      if (layerObj && layerObj.heads && layerObj.heads.length > headIdx) {
        const hMat = layerObj.heads[headIdx]?.attentionMatrix;
        if (hMat && hMat.length > 0) {
          mat = hMat;
        }
      }
      result.push({
        headIdx,
        matrix: mat,
        color: BERTVIZ_HEAD_COLORS[headIdx % BERTVIZ_HEAD_COLORS.length],
      });
    });
    return result;
  }, [allLayers, currentLayer, enabledHeads, fallbackMatrix]);

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        background: colors.canvas,
        padding: 16,
        borderRadius: 8,
        border: `1px solid ${colors.hairline}`,
        overflowX: 'auto',
      }}
    >
      <svg width={svgWidth} height={svgHeight} style={{ userSelect: 'none' }}>
        {/* Column Headers */}
        <text x={leftX - 8} y={14} textAnchor="end" fill={colors.inkMuted48} fontSize={10} fontWeight={700}>
          Query (Source)
        </text>
        <text x={rightX + 8} y={14} textAnchor="start" fill={colors.inkMuted48} fontSize={10} fontWeight={700}>
          Key (Target)
        </text>

        {/* Attention Arc Lines */}
        <g>
          {activeHeadMatrices.flatMap(({ headIdx, matrix, color }) => {
            const lines = [];
            for (let q = 0; q < tokens.length; q++) {
              for (let k = 0; k < tokens.length; k++) {
                const weight = matrix[q]?.[k] ?? 0;
                if (weight < attentionThreshold) continue;

                // Filtering condition
                const isQueryActive = activeQueryToken === null || activeQueryToken === q;
                const isKeyActive = activeKeyToken === null || activeKeyToken === k;
                const isHovered = hoveredToken === null || hoveredToken === q || hoveredToken === k;

                if (!isQueryActive || !isKeyActive || !isHovered) continue;

                const y1 = 28 + q * rowHeight + rowHeight / 2;
                const y2 = 28 + k * rowHeight + rowHeight / 2;
                const opacity = Math.min(0.9, Math.max(0.15, weight * 0.95));
                const strokeWidth = Math.max(1, weight * 5);

                const pathD = `M ${leftX} ${y1} C ${(leftX + rightX) / 2} ${y1}, ${(leftX + rightX) / 2} ${y2}, ${rightX} ${y2}`;

                lines.push(
                  <path
                    key={`line-h${headIdx}-q${q}-k${k}`}
                    d={pathD}
                    fill="none"
                    stroke={color}
                    strokeWidth={strokeWidth}
                    strokeOpacity={opacity}
                  />
                );
              }
            }
            return lines;
          })}
        </g>

        {/* Left Tokens (Queries) */}
        {tokens.map((tok, i) => {
          const y = 28 + i * rowHeight + rowHeight / 2;
          const isSelected = activeQueryToken === i;
          const isHovered = hoveredToken === i;

          return (
            <g
              key={`q-tok-${i}`}
              cursor="pointer"
              onMouseEnter={() => onHoverToken(i)}
              onMouseLeave={() => onHoverToken(null)}
              onClick={() => onSelectQueryToken(i)}
            >
              <rect
                x={0}
                y={28 + i * rowHeight + 2}
                width={leftX - 4}
                height={rowHeight - 4}
                rx={4}
                fill={isSelected ? colors.primary : isHovered ? colors.surfacePearl : 'transparent'}
                opacity={0.8}
              />
              <text
                x={leftX - 8}
                y={y}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={11}
                fontFamily="monospace"
                fontWeight={isSelected ? 700 : 500}
                fill={isSelected ? colors.onPrimary : colors.ink}
              >
                {tok}
              </text>
            </g>
          );
        })}

        {/* Right Tokens (Keys) */}
        {tokens.map((tok, j) => {
          const y = 28 + j * rowHeight + rowHeight / 2;
          const isSelected = activeKeyToken === j;
          const isHovered = hoveredToken === j;

          return (
            <g
              key={`k-tok-${j}`}
              cursor="pointer"
              onMouseEnter={() => onHoverToken(j)}
              onMouseLeave={() => onHoverToken(null)}
              onClick={() => onSelectKeyToken(j)}
            >
              <rect
                x={rightX + 4}
                y={28 + j * rowHeight + 2}
                width={tokenWidth - 4}
                height={rowHeight - 4}
                rx={4}
                fill={isSelected ? colors.primary : isHovered ? colors.surfacePearl : 'transparent'}
                opacity={0.8}
              />
              <text
                x={rightX + 8}
                y={y}
                textAnchor="start"
                dominantBaseline="middle"
                fontSize={11}
                fontFamily="monospace"
                fontWeight={isSelected ? 700 : 500}
                fill={isSelected ? colors.onPrimary : colors.ink}
              >
                {tok}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ============================================================================
// 2. BertViz Model View (Multi-Layer Multi-Head Mini Grid)
// ============================================================================

interface BertVizModelViewProps {
  tokens: string[];
  allLayers: Array<{ index: number; heads: Array<{ index: number; attentionMatrix: number[][] }> }>;
  selectedLayer: number;
  selectedHead: number;
  onSelectHead: (layer: number, head: number) => void;
}

function BertVizModelView({
  tokens,
  allLayers,
  selectedLayer,
  selectedHead,
  onSelectHead,
}: BertVizModelViewProps) {
  const numLayers = allLayers.length || 12;
  const numHeads = 12;
  const miniSize = 28;

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        background: colors.canvas,
        padding: 16,
        borderRadius: 8,
        border: `1px solid ${colors.hairline}`,
        overflowX: 'auto',
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.inkMuted48 }}>
        Model View (12 Layers × 12 Heads Matrix Grid) — Click head to view
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: `48px repeat(${numHeads}, ${miniSize}px)`, gap: 4, alignItems: 'center' }}>
        {/* Header Head Indices */}
        <div style={{ fontSize: 9, fontWeight: 700, color: colors.inkMuted48 }}>Layer</div>
        {Array.from({ length: numHeads }, (_, h) => (
          <div key={`h-idx-${h}`} style={{ fontSize: 9, fontWeight: 700, textAlign: 'center', color: colors.inkMuted48 }}>
            H{h}
          </div>
        ))}

        {/* Rows of Layers */}
        {Array.from({ length: numLayers }, (_, l) => (
          <React.Fragment key={`layer-row-${l}`}>
            <div style={{ fontSize: 10, fontWeight: 600, color: colors.ink }}>L{l}</div>
            {Array.from({ length: numHeads }, (_, h) => {
              const isSelected = selectedLayer === l && selectedHead === h;
              const mat = allLayers[l]?.heads[h]?.attentionMatrix || [];
              const maxVal = mat.length > 0 ? Math.max(...mat.flat()) : 0;

              return (
                <div
                  key={`cell-l${l}-h${h}`}
                  onClick={() => onSelectHead(l, h)}
                  title={`Layer ${l}, Head ${h} (Max Attention: ${maxVal.toFixed(2)})`}
                  style={{
                    width: miniSize,
                    height: miniSize,
                    borderRadius: 3,
                    cursor: 'pointer',
                    background: isSelected ? colors.primary : colors.surfacePearl,
                    border: `1px solid ${isSelected ? colors.primary : colors.hairline}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.1s ease',
                  }}
                >
                  <div
                    style={{
                      width: miniSize - 6,
                      height: miniSize - 6,
                      borderRadius: 2,
                      background: BERTVIZ_HEAD_COLORS[h % BERTVIZ_HEAD_COLORS.length],
                      opacity: maxVal > 0 ? Math.min(1.0, maxVal * 1.2) : 0.2,
                    }}
                  />
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// 3. BertViz Detailed Heatmap Matrix View
// ============================================================================

interface BertVizHeatmapMatrixProps {
  matrix: number[][];
  tokens: string[];
  hoveredToken: number | null;
  onHoverToken: (i: number | null) => void;
}

function BertVizHeatmapMatrix({
  matrix,
  tokens,
  hoveredToken,
  onHoverToken,
}: BertVizHeatmapMatrixProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; from: string; to: string; val: string } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const CELL_SIZE = 26;
  const LABEL_WIDTH = 80;
  const LABEL_HEIGHT = 20;

  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const W = LABEL_WIDTH + cols * CELL_SIZE;
  const H = LABEL_HEIGHT + rows * CELL_SIZE;

  const maxVal = useMemo(() => Math.max(...matrix.flat(), 1e-4), [matrix]);

  const getColor = (v: number) => {
    const t = Math.sqrt(Math.max(0, v) / maxVal);
    const r = Math.round(37 + (59 - 37) * (1 - t));
    const g = Math.round(99 + (130 - 99) * t);
    const b = Math.round(235 * t);
    return `rgba(${r},${g},${b},${Math.max(0.15, t)})`;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    ctx.fillStyle = colors.inkMuted48;
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < rows; i++) {
      ctx.fillText(tokens[i]?.slice(0, 8) ?? '', LABEL_WIDTH - 4, LABEL_HEIGHT + i * CELL_SIZE + CELL_SIZE / 2);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let j = 0; j < cols; j++) {
      ctx.fillText(tokens[j]?.slice(0, 8) ?? '', LABEL_WIDTH + j * CELL_SIZE + CELL_SIZE / 2, 2);
    }

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const x = LABEL_WIDTH + j * CELL_SIZE;
        const y = LABEL_HEIGHT + i * CELL_SIZE;
        ctx.fillStyle = getColor(matrix[i][j]);
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        if (hoveredToken === i || hoveredToken === j) {
          ctx.strokeStyle = colors.primary;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);
        }
      }
    }
  }, [matrix, tokens, hoveredToken, W, H, maxVal]);

  const handleMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const ci = Math.floor((my - LABEL_HEIGHT) / CELL_SIZE);
    const cj = Math.floor((mx - LABEL_WIDTH) / CELL_SIZE);
    if (ci >= 0 && ci < rows && cj >= 0 && cj < cols) {
      onHoverToken(ci);
      setTooltip({
        x: e.clientX - rect.left + 12,
        y: e.clientY - rect.top - 10,
        from: tokens[ci],
        to: tokens[cj],
        val: (matrix[ci]?.[cj] ?? 0).toFixed(4),
      });
    } else {
      onHoverToken(null);
      setTooltip(null);
    }
  };

  return (
    <div style={{ position: 'relative', overflowX: 'auto', background: colors.canvas, padding: 12, borderRadius: 8, border: `1px solid ${colors.hairline}` }}>
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        onMouseMove={handleMove}
        onMouseLeave={() => {
          onHoverToken(null);
          setTooltip(null);
        }}
      />
      {tooltip && (
        <div
          style={{
            position: 'absolute',
            left: tooltip.x,
            top: tooltip.y,
            background: colors.onDark,
            color: '#ffffff',
            padding: '4px 8px',
            borderRadius: 4,
            fontSize: 10,
            pointerEvents: 'none',
            fontFamily: 'monospace',
            zIndex: 10,
          }}
        >
          {tooltip.from} → {tooltip.to}: {tooltip.val}
        </div>
      )}
    </div>
  );
}
