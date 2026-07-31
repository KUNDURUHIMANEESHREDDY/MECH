import React, { useMemo, useState, useRef, useEffect } from 'react';

interface Props {
  matrix: number[][];
  tokens: string[];
  hoveredToken: number | null;
  onHoverToken: (i: number | null) => void;
}

const CELL_SIZE = 28;
const LABEL_WIDTH = 80;
const LABEL_HEIGHT = 20;

export function AttentionHeatmap({ matrix, tokens, hoveredToken, onHoverToken }: Props) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; from: string; to: string; val: string } | null>(null);
  const ref = useRef<HTMLCanvasElement>(null);

  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  const W = LABEL_WIDTH + cols * CELL_SIZE;
  const H = LABEL_HEIGHT + rows * CELL_SIZE;

  const maxVal = useMemo(() => Math.max(...matrix.flat()), [matrix]);

  const getColor = (v: number) => {
    // sqrt spreads low values so weak attention stays visible on dark theme
    const t = maxVal > 0 ? Math.sqrt(Math.max(0, v) / maxVal) : 0;
    const stops: [number, number, number][] = [
      [30, 41, 59],
      [14, 165, 233],
      [250, 204, 21],
    ];
    const scaled = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(scaled));
    const f = scaled - i;
    const r = Math.round(stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f);
    const g = Math.round(stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f);
    const b = Math.round(stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f);
    return `rgb(${r},${g},${b})`;
  };

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    // labels
    ctx.fillStyle = '#a0a0a0';
    ctx.font = '10px sans-serif';
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

    // cells
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const x = LABEL_WIDTH + j * CELL_SIZE;
        const y = LABEL_HEIGHT + i * CELL_SIZE;
        ctx.fillStyle = getColor(matrix[i][j]);
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        if (hoveredToken === i || hoveredToken === j) {
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);
        }
      }
    }
  }, [matrix, tokens, hoveredToken, W, H]);

  const handleMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = ref.current?.getBoundingClientRect();
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
        val: matrix[ci][cj].toFixed(3),
      });
    } else {
      onHoverToken(null);
      setTooltip(null);
    }
  };

  const handleLeave = () => {
    onHoverToken(null);
    setTooltip(null);
  };

  return (
    <div style={{ position: 'relative', overflow: 'auto' }}>
      <canvas
        ref={ref}
        width={W}
        height={H}
        onMouseMove={handleMove}
        onMouseLeave={handleLeave}
        style={{ cursor: 'crosshair', borderRadius: 4 }}
      />
      {tooltip && (
        <div style={{
          position: 'absolute',
          left: tooltip.x,
          top: tooltip.y,
          background: 'rgba(0,0,0,0.85)',
          color: '#fff',
          padding: '4px 8px',
          borderRadius: 4,
          fontSize: 11,
          pointerEvents: 'none',
          whiteSpace: 'nowrap',
          zIndex: 10,
        }}>
          {tooltip.from} &rarr; {tooltip.to}: {tooltip.val}
        </div>
      )}
    </div>
  );
}
