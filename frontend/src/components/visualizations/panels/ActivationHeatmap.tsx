import React, { useMemo, useRef, useEffect, useState } from 'react';
import { colors } from '../../../design/tokens/colors';
import { TokenActivationSpectrum } from './TokenActivationSpectrum';

interface Props {
  activations: number[];
  neuronIndex: number | null;
  onSelectNeuron: (i: number) => void;
  tokens?: string[];
  neuronTokenActivations?: (number[] | null | undefined)[];
}

const BASE_BAR_W = 16;
const BASE_BAR_H = 56;
const BAR_GAP = 4;
const LABEL_H = 14;
const PAD_X = 8;
const TOP_PAD = 18;
const ROW_BOTTOM_PAD = 8;
const BARS_AT_1X = 24;
const ZOOM_MIN = 0.3;
const ZOOM_MAX = 6;

const btnStyle: React.CSSProperties = {
  background: colors.surfacePearl,
  color: colors.inkMuted80,
  border: `1px solid ${colors.hairline}`,
  borderRadius: 4,
  width: 20,
  height: 20,
  padding: 0,
  cursor: 'pointer',
  fontSize: 13,
  lineHeight: '18px',
};

export function ActivationHeatmap({ activations, neuronIndex, onSelectNeuron, tokens, neuronTokenActivations }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [zoom, setZoom] = useState(1);
  const maxVal = useMemo(() => Math.max(...activations, 0.01), [activations]);

  const barsPerRow = Math.max(6, Math.round(BARS_AT_1X / zoom));
  const barW = BASE_BAR_W * zoom;
  const barH = BASE_BAR_H * zoom;
  const rowH = TOP_PAD + barH + LABEL_H + ROW_BOTTOM_PAD;
  const rows = Math.max(1, Math.ceil(activations.length / barsPerRow));
  const W = PAD_X * 2 + barsPerRow * (barW + BAR_GAP);
  const H = rows * rowH;

  const selTokenActivations = neuronIndex !== null ? neuronTokenActivations?.[neuronIndex] : undefined;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    for (let i = 0; i < activations.length; i++) {
      const row = Math.floor(i / barsPerRow);
      const col = i % barsPerRow;
      const x = PAD_X + col * (barW + BAR_GAP);
      const rowTop = row * rowH;
      const v = activations[i];
      const barHeight = Math.max(2, (v / maxVal) * barH);
      const barTop = rowTop + TOP_PAD + (barH - barHeight);
      const selected = neuronIndex === i;

      ctx.fillStyle = selected ? colors.primary : colors.surfacePearl;
      ctx.fillRect(x, barTop, barW, barHeight);

      const t = v / maxVal;
      const r = Math.round(60 + t * 195);
      const g = Math.round(40 + (1 - t) * 80);
      const b = Math.round(200);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x, barTop, barW, barHeight);

      if (selected) {
        ctx.strokeStyle = colors.warning;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, barTop, barW, barHeight);
      }

      ctx.fillStyle = colors.inkMuted48;
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(`n${i}`, x + barW / 2, rowTop + TOP_PAD + barH + LABEL_H);

      ctx.fillStyle = colors.bodyMuted;
      ctx.fillText(v.toFixed(2), x + barW / 2, rowTop + TOP_PAD - 4);
    }
  }, [activations, maxVal, neuronIndex, zoom, W, H, barW, barH, rowH, barsPerRow]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setZoom(z => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z * (e.deltaY < 0 ? 1.2 : 1 / 1.2))));
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, []);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const col = Math.floor((mx - PAD_X) / (barW + BAR_GAP));
    const row = Math.floor(my / rowH);
    const idx = row * barsPerRow + col;
    if (idx >= 0 && idx < activations.length) {
      onSelectNeuron(idx);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <span style={{ fontSize: 10, color: colors.inkMuted48 }}>Zoom</span>
        <button type="button" onClick={() => setZoom(z => Math.max(ZOOM_MIN, z - 0.1))} style={btnStyle}>-</button>
        <span style={{ fontSize: 10, fontFamily: 'monospace', minWidth: 36, textAlign: 'center', color: colors.bodyMuted }}>{Math.round(zoom * 100)}%</span>
        <button type="button" onClick={() => setZoom(z => Math.min(ZOOM_MAX, z + 0.1))} style={btnStyle}>+</button>
        <span style={{ fontSize: 10, color: colors.inkMuted48 }}>scroll to zoom</span>
      </div>
      <canvas
        ref={ref}
        width={W}
        height={H}
        onClick={handleClick}
        style={{ cursor: 'pointer', borderRadius: 4 }}
      />
      {neuronIndex !== null && tokens && tokens.length > 0 && selTokenActivations && selTokenActivations.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <TokenActivationSpectrum tokens={tokens} activations={selTokenActivations} />
        </div>
      )}
    </div>
  );
}
