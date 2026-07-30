import React, { useMemo, useRef, useEffect } from 'react';

interface Props {
  activations: number[];
  neuronIndex: number | null;
  onSelectNeuron: (i: number) => void;
}

const BAR_HEIGHT = 22;
const BAR_GAP = 4;
const LABEL_W = 60;

export function ActivationHeatmap({ activations, neuronIndex, onSelectNeuron }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const maxVal = useMemo(() => Math.max(...activations, 0.01), [activations]);
  const H = activations.length * (BAR_HEIGHT + BAR_GAP);
  const W = LABEL_W + 160;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    for (let i = 0; i < activations.length; i++) {
      const y = i * (BAR_HEIGHT + BAR_GAP);
      const v = activations[i];
      const barW = (v / maxVal) * 140;

      // label
      ctx.fillStyle = '#a0a0a0';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`n${i}`, LABEL_W - 6, y + BAR_HEIGHT / 2);

      // bar bg
      ctx.fillStyle = neuronIndex === i ? '#3a5a8a' : '#1e1e2e';
      ctx.fillRect(LABEL_W, y, 140, BAR_HEIGHT);

      // bar fill
      const t = v / maxVal;
      const r = Math.round(60 + t * 195);
      const g = Math.round(40 + (1 - t) * 80);
      const b = Math.round(200);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(LABEL_W, y, barW, BAR_HEIGHT);

      // highlight border
      if (neuronIndex === i) {
        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth = 2;
        ctx.strokeRect(LABEL_W, y, 140, BAR_HEIGHT);
      }

      // value text
      ctx.fillStyle = '#ccc';
      ctx.textAlign = 'left';
      ctx.fillText(v.toFixed(3), LABEL_W + barW + 6, y + BAR_HEIGHT / 2);
    }
  }, [activations, maxVal, neuronIndex, W, H]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const my = e.clientY - rect.top;
    const idx = Math.floor(my / (BAR_HEIGHT + BAR_GAP));
    if (idx >= 0 && idx < activations.length) {
      onSelectNeuron(idx);
    }
  };

  return (
    <canvas
      ref={ref}
      width={W}
      height={H}
      onClick={handleClick}
      style={{ cursor: 'pointer', borderRadius: 4 }}
    />
  );
}
