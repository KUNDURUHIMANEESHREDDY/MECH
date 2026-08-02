import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import { NeuronPoint, NeuronUMapTheme, ProjectedPoint, SelectionRect, Viewport } from './types';
import { projectNeurons } from './projection';
import { exportSceneToDataURL, screenToWorld } from './renderer';

const VIEWPORT_CACHE = new Map<string, Viewport>();

const MIN_ZOOM = 1;
const MAX_ZOOM = 60;
const NEIGHBOR_COUNT = 6;
const CELL = 0.02;
const ANIM_MS = 320;

function cacheKey(points: NeuronPoint[]): string {
  const dim = points[0]?.embedding.length ?? 0;
  return `${points.length}|${dim}|${points[0]?.layer ?? 0}-${points[points.length - 1]?.layer ?? 0}`;
}

function clampViewport(vp: Viewport): Viewport {
  const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, vp.k));
  const x = Math.min(Math.max(vp.x, 0), Math.max(0, 1 - 1 / k));
  const y = Math.min(Math.max(vp.y, 0), Math.max(0, 1 - 1 / k));
  return { x, y, k };
}

function unionFind(size: number): { find: (i: number) => number; union: (a: number, b: number) => void; count: () => number } {
  const parent = Array.from({ length: size }, (_, i) => i);
  const rank = new Array(size).fill(0);
  const find = (i: number): number => {
    let root = i;
    while (parent[root] !== root) root = parent[root];
    while (parent[i] !== i) {
      const next = parent[i];
      parent[i] = root;
      i = next;
    }
    return root;
  };
  const union = (a: number, b: number) => {
    const ra = find(a);
    const rb = find(b);
    if (ra === rb) return;
    if (rank[ra] < rank[rb]) parent[ra] = rb;
    else if (rank[ra] > rank[rb]) parent[rb] = ra;
    else {
      parent[rb] = ra;
      rank[ra]++;
    }
  };
  const count = () => {
    const roots = new Set<number>();
    for (let i = 0; i < size; i++) roots.add(find(i));
    return roots.size;
  };
  return { find, union, count };
}

function buildGrid(coords: ProjectedPoint[]): { cols: number; rows: number; cells: number[][] } {
  const cols = Math.ceil(1 / CELL) + 1;
  const rows = Math.ceil(1 / CELL) + 1;
  const cells: number[][] = Array.from({ length: cols * rows }, () => []);
  for (let i = 0; i < coords.length; i++) {
    const c = coords[i];
    const cx = Math.min(cols - 1, Math.max(0, Math.floor(c.x / CELL)));
    const cy = Math.min(rows - 1, Math.max(0, Math.floor(c.y / CELL)));
    cells[cy * cols + cx].push(i);
  }
  return { cols, rows, cells };
}

function computeNeighbors(coords: ProjectedPoint[]): Map<number, number[]> {
  const { cols, rows, cells } = buildGrid(coords);
  const result = new Map<number, number[]>();
  for (let i = 0; i < coords.length; i++) {
    const c = coords[i];
    const cx = Math.min(cols - 1, Math.max(0, Math.floor(c.x / CELL)));
    const cy = Math.min(rows - 1, Math.max(0, Math.floor(c.y / CELL)));
    const candidates: number[] = [];
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
        const cell = cells[ny * cols + nx];
        for (let j = 0; j < cell.length; j++) {
          const idx = cell[j];
          if (idx !== i) candidates.push(idx);
        }
      }
    }
    candidates.sort((a, b) => {
      const da = (coords[a].x - c.x) ** 2 + (coords[a].y - c.y) ** 2;
      const db = (coords[b].x - c.x) ** 2 + (coords[b].y - c.y) ** 2;
      return da - db;
    });
    result.set(i, candidates.slice(0, NEIGHBOR_COUNT));
  }
  return result;
}

export interface UseNeuronUMAPOptions {
  points: NeuronPoint[];
  selectedId: string | null;
  onSelectNeuron?: (id: string | null) => void;
  focusSearchRef?: { current: HTMLInputElement | null };
}

export function useNeuronUMAP({ points, selectedId, onSelectNeuron, focusSearchRef }: UseNeuronUMAPOptions) {
  const coords = useMemo(() => projectNeurons(points), [points]);
  const key = useMemo(() => cacheKey(points), [points]);

  const animRef = useRef<ProjectedPoint[]>(coords);
  const [viewport, setViewport] = useState<Viewport>(() => VIEWPORT_CACHE.get(key) ?? { x: 0, y: 0, k: 1 });
  const viewportRef = useRef(viewport);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const hoveredRef = useRef<number | null>(null);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const selectedRef = useRef<Set<number>>(new Set());
  const [selectionRect, setSelectionRect] = useState<SelectionRect | null>(null);
  const selectionRectRef = useRef<SelectionRect | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [layerFilter, setLayerFilter] = useState<number | null>(null);
  const [headFilter, setHeadFilter] = useState<number | null>(null);
  const [threshold, setThreshold] = useState(0.02);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 600, h: 400 });
  const containerRef = useRef<HTMLDivElement | null>(null);

  const panRef = useRef<{ start: { x: number; y: number }; vp0: Viewport } | null>(null);
  const selectRef = useRef<{ x0: number; y0: number; additive: boolean } | null>(null);
  const [interactionMode, setInteractionMode] = useState<'pan' | 'select'>('pan');

  const maxAbsAct = useMemo(() => {
    let m = 0;
    for (const p of points) m = Math.max(m, Math.abs(p.activation));
    return m;
  }, [points]);

  const hidden = useMemo(() => {
    const h = new Set<number>();
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (layerFilter !== null && p.layer !== layerFilter) h.add(i);
      if (headFilter !== null && p.head !== undefined && p.head !== headFilter) h.add(i);
      if (maxAbsAct > 0 && Math.abs(p.activation) < threshold * maxAbsAct) h.add(i);
    }
    return h;
  }, [points, layerFilter, headFilter, threshold, maxAbsAct]);

  const colorClasses = useMemo(() => {
    const arr = new Uint8Array(points.length);
    for (let i = 0; i < points.length; i++) {
      const a = points[i].activation;
      if (maxAbsAct <= 0) {
        arr[i] = 0;
      } else if (a >= threshold * maxAbsAct) arr[i] = 1;
      else if (a <= -threshold * maxAbsAct) arr[i] = 2;
      else arr[i] = 0;
    }
    return arr;
  }, [points, threshold, maxAbsAct]);

  const neighbors = useMemo(() => computeNeighbors(coords), [coords]);

  const stats = useMemo(() => {
    let positive = 0;
    let negative = 0;
    let neutral = 0;
    let sum = 0;
    let visible = 0;
    for (let i = 0; i < points.length; i++) {
      if (hidden.has(i)) continue;
      visible++;
      const a = points[i].activation;
      sum += Math.abs(a);
      if (colorClasses[i] === 1) positive++;
      else if (colorClasses[i] === 2) negative++;
      else neutral++;
    }
    const uf = unionFind(points.length);
    for (let i = 0; i < points.length; i++) {
      if (hidden.has(i)) continue;
      const nbrs = neighbors.get(i) ?? [];
      for (const n of nbrs) {
        if (!hidden.has(n)) uf.union(i, n);
      }
    }
    const rootSet = new Set<number>();
    for (let i = 0; i < points.length; i++) {
      if (!hidden.has(i)) rootSet.add(uf.find(i));
    }
    return {
      total: points.length,
      positive,
      negative,
      neutral,
      active: positive + negative,
      avgActivation: visible > 0 ? sum / visible : 0,
      clusters: rootSet.size,
    };
  }, [points, hidden, colorClasses, neighbors]);

  useEffect(() => {
    VIEWPORT_CACHE.set(key, viewport);
    viewportRef.current = viewport;
  }, [key, viewport]);

  useEffect(() => {
    if (selectedId !== null && selectedId !== undefined) {
      const idx = points.findIndex(p => p.id === selectedId);
      const set = new Set<number>();
      if (idx >= 0) set.add(idx);
      selectedRef.current = set;
      setSelectedIndices(set);
    }
  }, [selectedId, points]);

  useEffect(() => {
    const from = animRef.current;
    const to = coords;
    if (from.length === to.length && from.length === 0) return;
    if (from === to) return;
    if (from.length !== to.length) {
      animRef.current = to;
      return;
    }
    let raf = 0;
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / ANIM_MS);
      const eased = 1 - (1 - t) * (1 - t);
      const next = new Array<ProjectedPoint>(from.length);
      for (let i = 0; i < from.length; i++) {
        next[i] = {
          x: from[i].x + (to[i].x - from[i].x) * eased,
          y: from[i].y + (to[i].y - from[i].y) * eased,
        };
      }
      animRef.current = next;
      if (t < 1) raf = requestAnimationFrame(step);
      else animRef.current = to;
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [coords]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const rect = entries[0]?.contentRect;
      if (rect) {
        setSize(s => (s.w === rect.width && s.h === rect.height ? s : { w: rect.width, h: rect.height }));
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const fitView = useCallback(() => {
    const next = { x: 0, y: 0, k: 1 };
    setViewport(next);
  }, []);

  const zoomAt = useCallback((factor: number, cx?: number, cy?: number) => {
    setViewport(prev => {
      const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev.k * factor));
      const w = size.w;
      const h = size.h;
      const px = cx ?? w / 2;
      const py = cy ?? h / 2;
      const wx = prev.x + px / (w * prev.k);
      const wy = prev.y + py / (h * prev.k);
      const x = wx - px / (w * k);
      const y = wy - py / (h * k);
      return clampViewport({ x, y, k });
    });
  }, [size.w, size.h]);

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.25 : 1 / 1.25;
    zoomAt(factor, cx, cy);
  }, [zoomAt]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [handleWheel]);

  const pointAt = useCallback((sx: number, sy: number): number | null => {
    if (size.w <= 0 || size.h <= 0) return null;
    const world = screenToWorld(sx, sy, viewportRef.current, size.w, size.h);
    const { cols, rows, cells } = buildGrid(coords);
    const cx = Math.min(cols - 1, Math.max(0, Math.floor(world.x / CELL)));
    const cy = Math.min(rows - 1, Math.max(0, Math.floor(world.y / CELL)));
    const screenRadius = 10 / viewportRef.current.k;
    let best: number | null = null;
    let bestD = screenRadius * screenRadius;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
        const cell = cells[ny * cols + nx];
        for (let j = 0; j < cell.length; j++) {
          const idx = cell[j];
          const p = coords[idx];
          const dxp = p.x - world.x;
          const dyp = p.y - world.y;
          const d = dxp * dxp + dyp * dyp;
          if (d < bestD) {
            bestD = d;
            best = idx;
          }
        }
      }
    }
    return best;
  }, [coords, size.w, size.h]);

  const setHover = useCallback((idx: number | null) => {
    hoveredRef.current = idx;
    setHoveredIdx(idx);
  }, []);

  const onPointerLeave = useCallback(() => {
    setHover(null);
  }, [setHover]);

  const onPointerDown = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    if (interactionMode === 'select') {
      selectRef.current = { x0: sx, y0: sy, additive: e.shiftKey || e.ctrlKey || e.metaKey };
      selectionRectRef.current = { x0: sx, y0: sy, x1: sx, y1: sy };
      setSelectionRect(selectionRectRef.current);
    } else {
      panRef.current = { start: { x: sx, y: sy }, vp0: { ...viewportRef.current } };
    }
  }, [interactionMode]);

  const onPointerMove = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    if (selectRef.current) {
      const next = { x0: selectRef.current.x0, y0: selectRef.current.y0, x1: sx, y1: sy };
      selectionRectRef.current = next;
      setSelectionRect(next);
      return;
    }

    if (panRef.current) {
      const dx = sx - panRef.current.start.x;
      const dy = sy - panRef.current.start.y;
      const k = panRef.current.vp0.k;
      const x = panRef.current.vp0.x - dx / (size.w * k);
      const y = panRef.current.vp0.y - dy / (size.h * k);
      setViewport(clampViewport({ x, y, k }));
      return;
    }

    const idx = pointAt(sx, sy);
    if (idx !== hoveredRef.current) setHover(idx);
  }, [pointAt, size.w, size.h]);

  const onPointerUp = useCallback((e: ReactPointerEvent<HTMLElement>) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;

    if (selectRef.current) {
      const sel = selectRef.current;
      selectRef.current = null;
      selectionRectRef.current = null;
      setSelectionRect(null);
      const dx = sx - sel.x0;
      const dy = sy - sel.y0;
      if (Math.abs(dx) < 4 && Math.abs(dy) < 4) {
        const idx = pointAt(sx, sy);
        const next = new Set(selectedRef.current);
        if (idx !== null) {
          if (sel.additive) {
            if (next.has(idx)) next.delete(idx);
            else next.add(idx);
          } else {
            next.clear();
            next.add(idx);
          }
        } else if (!sel.additive) {
          next.clear();
        }
        selectedRef.current = next;
        setSelectedIndices(next);
        if (idx !== null && onSelectNeuron) onSelectNeuron(points[idx].id);
        if (idx === null && onSelectNeuron) onSelectNeuron(null);
        return;
      }
      const w0 = screenToWorld(Math.min(sel.x0, sx), Math.min(sel.y0, sy), viewportRef.current, size.w, size.h);
      const w1 = screenToWorld(Math.max(sel.x0, sx), Math.max(sel.y0, sy), viewportRef.current, size.w, size.h);
      const next = sel.additive ? new Set(selectedRef.current) : new Set<number>();
      for (let i = 0; i < coords.length; i++) {
        const p = coords[i];
        if (hidden.has(i)) continue;
        if (p.x >= w0.x && p.x <= w1.x && p.y >= w0.y && p.y <= w1.y) next.add(i);
      }
      selectedRef.current = next;
      setSelectedIndices(next);
      return;
    }

    if (panRef.current) {
      panRef.current = null;
    }
  }, [coords, hidden, pointAt, size.w, size.h, onSelectNeuron, points]);

  const clearSelection = useCallback(() => {
    selectedRef.current = new Set();
    setSelectedIndices(new Set());
    if (onSelectNeuron) onSelectNeuron(null);
  }, [onSelectNeuron]);

  const selectById = useCallback((id: string) => {
    const idx = points.findIndex(p => p.id === id);
    if (idx < 0) return;
    selectedRef.current = new Set([idx]);
    setSelectedIndices(new Set([idx]));
    if (onSelectNeuron) onSelectNeuron(id);
  }, [points, onSelectNeuron]);

  const exportPNG = useCallback(
    (theme: NeuronUMapTheme) => {
      const url = exportSceneToDataURL(size.w, size.h, theme, coords, colorClasses);
      if (!url) return;
      const a = document.createElement('a');
      a.href = url;
      a.download = 'neuron-umap.png';
      a.click();
    },
    [coords, colorClasses, size.w, size.h],
  );

  const searchResults = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return [];
    const results: { id: string; idx: number }[] = [];
    for (let i = 0; i < points.length && results.length < 8; i++) {
      const id = points[i].id.toLowerCase();
      if (id.includes(q)) results.push({ id: points[i].id, idx: i });
    }
    return results;
  }, [searchQuery, points]);

  const layers = useMemo(() => {
    const set = new Set<number>();
    for (const p of points) set.add(p.layer);
    return [...set].sort((a, b) => a - b);
  }, [points]);

  const heads = useMemo(() => {
    const set = new Set<number>();
    for (const p of points) {
      if (p.head !== undefined) set.add(p.head);
    }
    return [...set].sort((a, b) => a - b);
  }, [points]);

  const onKeyDown = useCallback((e: ReactKeyboardEvent<HTMLElement>) => {
    if (e.key === 'Escape') {
      selectionRectRef.current = null;
      setSelectionRect(null);
      selectRef.current = null;
      clearSelection();
      setHover(null);
    } else if (e.key === 'f' || e.key === 'F') {
      fitView();
    } else if (e.key === '+' || e.key === '=') {
      zoomAt(1.3);
    } else if (e.key === '-') {
      zoomAt(1 / 1.3);
    } else if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      focusSearchRef.current?.focus();
    }
  }, [clearSelection, fitView, zoomAt, focusSearchRef, setHover]);

  const hoveredPoint = hoveredIdx !== null ? points[hoveredIdx] : null;
  const selectedPoints = useMemo(() => {
    const arr: NeuronPoint[] = [];
    selectedIndices.forEach(i => {
      if (i >= 0 && i < points.length) arr.push(points[i]);
    });
    return arr;
  }, [points, selectedIndices]);

  return {
    containerRef,
    size,
    viewport,
    animRef,
    hoveredIdx,
    hoveredPoint,
    selectedIndices,
    selectedPoints,
    selectionRect,
    colorClasses,
    neighbors,
    stats,
    hidden,
    searchQuery,
    setSearchQuery,
    searchResults,
    layerFilter,
    setLayerFilter,
    layers,
    headFilter,
    setHeadFilter,
    heads,
    threshold,
    setThreshold,
    interactionMode,
    setInteractionMode,
    maxAbsAct,
    fitView,
    zoomAt,
    exportPNG,
    clearSelection,
    selectById,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerLeave,
    onKeyDown,
  };
}
