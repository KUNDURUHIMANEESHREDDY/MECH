import { ProjectedPoint, SelectionRect, Viewport, NeuronUMapTheme } from './types';

export interface SceneDrawOptions {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
  theme: NeuronUMapTheme;
  coords: ProjectedPoint[];
  colorClasses: Uint8Array;
  viewport: Viewport;
  hovered: number | null;
  selected: Set<number>;
  neighbors: Map<number, number[]> | null;
  selectionRect: SelectionRect | null;
  hidden: Set<number> | null;
  showLabels?: boolean;
}

export function worldToScreen(p: ProjectedPoint, vp: Viewport, w: number, h: number): { x: number; y: number } {
  return { x: (p.x - vp.x) * w * vp.k, y: (p.y - vp.y) * h * vp.k };
}

export function screenToWorld(sx: number, sy: number, vp: Viewport, w: number, h: number): { x: number; y: number } {
  return { x: vp.x + sx / (w * vp.k), y: vp.y + sy / (h * vp.k) };
}

function pointRadius(k: number, base: number): number {
  return Math.min(9, base * Math.sqrt(k));
}

export function drawScene(opts: SceneDrawOptions): void {
  const { ctx, w, h, theme, coords, colorClasses, viewport, hovered, selected, neighbors, selectionRect, hidden } = opts;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);

  const gridStep = 0.1;
  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = 1;
  if (viewport.k <= 6) {
    ctx.beginPath();
    const x0 = Math.floor(viewport.x / gridStep) * gridStep;
    const x1 = viewport.x + 1 / viewport.k;
    const y0 = Math.floor(viewport.y / gridStep) * gridStep;
    const y1 = viewport.y + 1 / viewport.k;
    for (let gx = x0; gx <= x1 + gridStep; gx += gridStep) {
      const sx = (gx - viewport.x) * w * viewport.k;
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, h);
    }
    for (let gy = y0; gy <= y1 + gridStep; gy += gridStep) {
      const sy = (gy - viewport.y) * h * viewport.k;
      ctx.moveTo(0, sy);
      ctx.lineTo(w, sy);
    }
    ctx.stroke();
  }

  const pad = 24;
  const xMinScreen = -pad;
  const yMinScreen = -pad;
  const xMaxScreen = w + pad;
  const yMaxScreen = h + pad;

  const k = viewport.k;
  const base = 2.4;
  const radius = pointRadius(k, base);

  for (let i = 0; i < coords.length; i++) {
    if (hidden && hidden.has(i)) continue;
    const c = coords[i];
    const sx = (c.x - viewport.x) * w * k;
    const sy = (c.y - viewport.y) * h * k;
    if (sx < xMinScreen || sx > xMaxScreen || sy < yMinScreen || sy > yMaxScreen) continue;

    const cls = colorClasses[i];
    const isHovered = hovered === i;
    const isSelected = selected.has(i);

    let fill = theme.gray;
    if (cls === 1) fill = theme.green;
    else if (cls === 2) fill = theme.red;
    if (isSelected) fill = theme.accent;

    ctx.globalAlpha = 1;
    ctx.fillStyle = fill;
    if (radius <= 2.5) {
      ctx.fillRect(sx - radius * 0.6, sy - radius * 0.6, radius * 1.2, radius * 1.2);
    } else {
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (neighbors && hovered !== null) {
    const nbrs = neighbors.get(hovered);
    if (nbrs) {
      const hc = coords[hovered];
      const hx = (hc.x - viewport.x) * w * k;
      const hy = (hc.y - viewport.y) * h * k;
      ctx.strokeStyle = theme.hover;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const n of nbrs) {
        const nc = coords[n];
        const nx = (nc.x - viewport.x) * w * k;
        const ny = (nc.y - viewport.y) * h * k;
        if (nx < xMinScreen || nx > xMaxScreen || ny < yMinScreen || ny > yMaxScreen) continue;
        ctx.moveTo(hx, hy);
        ctx.lineTo(nx, ny);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  if (hovered !== null) {
    const hc = coords[hovered];
    const hx = (hc.x - viewport.x) * w * k;
    const hy = (hc.y - viewport.y) * h * k;
    ctx.strokeStyle = theme.hover;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(hx, hy, radius + 3.5, 0, Math.PI * 2);
    ctx.stroke();
  }

  for (const i of selected) {
    const c = coords[i];
    const sx = (c.x - viewport.x) * w * k;
    const sy = (c.y - viewport.y) * h * k;
    if (sx < xMinScreen || sx > xMaxScreen || sy < yMinScreen || sy > yMaxScreen) continue;
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, radius + 3, 0, Math.PI * 2);
    ctx.stroke();
  }

  if (selectionRect) {
    const rx = Math.min(selectionRect.x0, selectionRect.x1);
    const ry = Math.min(selectionRect.y0, selectionRect.y1);
    const rw = Math.abs(selectionRect.x1 - selectionRect.x0);
    const rh = Math.abs(selectionRect.y1 - selectionRect.y0);
    ctx.fillStyle = theme.accent;
    ctx.globalAlpha = 0.12;
    ctx.fillRect(rx, ry, rw, rh);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = theme.accent;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
  }
}

export function drawMinimap(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  coords: ProjectedPoint[],
  colorClasses: Uint8Array,
  theme: NeuronUMapTheme,
  viewport: Viewport,
): void {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < coords.length; i++) {
    const c = coords[i];
    const cls = colorClasses[i];
    let fill = theme.gray;
    if (cls === 1) fill = theme.green;
    else if (cls === 2) fill = theme.red;
    ctx.fillStyle = fill;
    ctx.globalAlpha = 0.7;
    ctx.fillRect(c.x * w - 1, c.y * h - 1, 2, 2);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = theme.accent;
  ctx.lineWidth = 1;
  ctx.strokeRect(viewport.x * w, viewport.y * h, w / viewport.k, h / viewport.k);
}

export function exportSceneToDataURL(
  w: number,
  h: number,
  theme: NeuronUMapTheme,
  coords: ProjectedPoint[],
  colorClasses: Uint8Array,
): string {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  const scale = 2;
  canvas.width = w * scale;
  canvas.height = h * scale;
  ctx.scale(scale, scale);
  drawScene({
    ctx,
    w,
    h,
    theme,
    coords,
    colorClasses,
    viewport: { x: 0, y: 0, k: 1 },
    hovered: null,
    selected: new Set(),
    neighbors: null,
    selectionRect: null,
    hidden: null,
  });
  return canvas.toDataURL('image/png');
}
