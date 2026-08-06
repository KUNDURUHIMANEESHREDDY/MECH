import { useCallback, useEffect, useMemo, useRef } from 'react';
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import { BoxSelect, Download, Hand, Loader2, Maximize, Search, ZoomIn, ZoomOut } from 'lucide-react';
import { colors } from '../../../design/tokens/colors';
import { NeuronUMapProps, NeuronUMapTheme } from './types';
import { useNeuronUMAP } from './useNeuronUMAP';
import { drawMinimap, drawScene } from './renderer';
import { ConnectedNeuron, NeuronDetailPanel } from './NeuronDetailPanel';

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgba(hex: string, a: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

const LIGHT_THEME: NeuronUMapTheme = {
  bg: colors.canvasParchment,
  grid: colors.hairline,
  text: colors.ink,
  muted: colors.inkMuted48,
  cardBg: colors.canvas,
  border: colors.border,
  green: colors.success,
  red: colors.danger,
  gray: colors.inkMuted48,
  accent: colors.primary,
  hover: colors.ink,
  tooltipBg: rgba(colors.ink, 0.95),
};

const MINIMAP_W = 140;
const MINIMAP_H = 100;

export function NeuronUMAP({
  points,
  selectedId = null,
  onSelectNeuron,
  height = 480,
  loading = false,
  title = 'Neuron Map',
}: NeuronUMapProps) {
  const theme = LIGHT_THEME;

  const searchRef = useRef<HTMLInputElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const minimapRef = useRef<HTMLCanvasElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const mouseRef = useRef({ x: 0, y: 0 });

  const {
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
  } = useNeuronUMAP({ points, selectedId, onSelectNeuron, focusSearchRef: searchRef });

  const drawRefs = useRef({
    viewport,
    hoveredIdx,
    selectedIndices,
    selectionRect,
    hidden,
    neighbors,
    colorClasses,
  });
  drawRefs.current = { viewport, hoveredIdx, selectedIndices, selectionRect, hidden, neighbors, colorClasses };

  useEffect(() => {
    const canvas = canvasRef.current;
    const minimap = minimapRef.current;
    const tooltip = tooltipRef.current;
    if (!canvas || !minimap) return;
    const ctx = canvas.getContext('2d');
    const mctx = minimap.getContext('2d');
    if (!ctx || !mctx) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(size.w * dpr));
    canvas.height = Math.max(1, Math.round(size.h * dpr));
    minimap.width = Math.max(1, Math.round(MINIMAP_W * dpr));
    minimap.height = Math.max(1, Math.round(MINIMAP_H * dpr));

    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const d = drawRefs.current;
      const coords = animRef.current;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawScene({
        ctx,
        w: size.w,
        h: size.h,
        theme,
        coords,
        colorClasses: d.colorClasses,
        viewport: d.viewport,
        hovered: d.hoveredIdx,
        selected: d.selectedIndices,
        neighbors: d.neighbors,
        selectionRect: d.selectionRect,
        hidden: d.hidden,
      });
      mctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawMinimap(mctx, MINIMAP_W, MINIMAP_H, coords, d.colorClasses, theme, d.viewport);

      if (tooltip) {
        if (d.hoveredIdx !== null) {
          tooltip.style.display = 'block';
          const m = mouseRef.current;
          let left = m.x + 14;
          let top = m.y + 14;
          if (left + tooltip.offsetWidth > size.w) left = m.x - tooltip.offsetWidth - 10;
          if (top + tooltip.offsetHeight > size.h) top = m.y - tooltip.offsetHeight - 10;
          tooltip.style.transform = `translate(${left}px, ${top}px)`;
        } else {
          tooltip.style.display = 'none';
        }
      }
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [size.w, size.h, theme, animRef]);

  useEffect(() => {
    const onDocMouseDown = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        clearSelection();
      }
    };
    document.addEventListener('mousedown', onDocMouseDown);
    return () => document.removeEventListener('mousedown', onDocMouseDown);
  }, [clearSelection, containerRef]);

  useEffect(() => {
    containerRef.current?.focus({ preventScroll: true });
  }, [containerRef]);

  const handlePointerMove = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (rect) mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      onPointerMove(e);
    },
    [containerRef, onPointerMove],
  );

  const handleKeyDown = useCallback(
    (e: ReactKeyboardEvent<HTMLElement>) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      onKeyDown(e);
    },
    [onKeyDown],
  );

  const primaryIdx = selectedIndices.size > 0 ? [...selectedIndices][0] : null;
  const primaryPoint = primaryIdx !== null && primaryIdx < points.length ? points[primaryIdx] : null;

  const connected: ConnectedNeuron[] = (() => {
    if (primaryIdx === null) return [];
    const nbrs = neighbors.get(primaryIdx) ?? [];
    const cur = animRef.current;
    const hc = cur[primaryIdx];
    if (!hc) return [];
    const arr: ConnectedNeuron[] = [];
    for (const n of nbrs) {
      if (hidden.has(n)) continue;
      const nc = cur[n];
      if (!nc) continue;
      arr.push({ point: points[n], dist: Math.hypot(nc.x - hc.x, nc.y - hc.y) });
    }
    return arr;
  })();

  const btn: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: 5,
    padding: '4px 8px',
    borderRadius: 6,
    border: `1px solid ${theme.border}`,
    background: theme.cardBg,
    color: theme.text,
    fontSize: 12,
    cursor: 'pointer',
  };

  const btnActive: CSSProperties = { ...btn, borderColor: theme.accent, color: theme.accent };

  const selectStyle: CSSProperties = {
    padding: '3px 6px',
    borderRadius: 6,
    border: `1px solid ${theme.border}`,
    background: theme.cardBg,
    color: theme.text,
    fontSize: 12,
  };

  const inputStyle: CSSProperties = {
    ...selectStyle,
    width: 130,
    outline: 'none',
  };

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onPointerDown={onPointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={onPointerLeave}
      onDoubleClick={fitView}
      style={{
        position: 'relative',
        width: '100%',
        height,
        overflow: 'hidden',
        borderRadius: 10,
        border: `1px solid ${theme.border}`,
        background: theme.bg,
        color: theme.text,
        userSelect: 'none',
        outline: 'none',
        cursor: interactionMode === 'pan' ? 'grab' : 'crosshair',
      }}
    >
      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', display: 'block' }} />

      <div
        onPointerDown={e => e.stopPropagation()}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          display: 'flex',
          flexWrap: 'wrap',
          gap: 6,
          alignItems: 'center',
          padding: '6px 8px',
          background: theme.cardBg,
          borderBottom: `1px solid ${theme.border}`,
          zIndex: 6,
        }}
      >
        <span style={{ fontSize: 12, fontWeight: 600, marginRight: 4 }}>{title}</span>

        <div style={{ position: 'relative' }}>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={12} style={{ position: 'absolute', left: 6, color: theme.muted }} />
            <input
              ref={searchRef}
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search neuron ID… (Ctrl+K)"
              style={{ ...inputStyle, paddingLeft: 20 }}
            />
          </div>
          {searchQuery.trim() !== '' && (
            <div
              style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                minWidth: 180,
                background: theme.cardBg,
                border: `1px solid ${theme.border}`,
                borderRadius: 6,
                boxShadow: '0 6px 18px rgba(0,0,0,0.16)',
                zIndex: 8,
                overflow: 'hidden',
              }}
            >
              {searchResults.length === 0 ? (
                <div style={{ padding: '6px 10px', fontSize: 11, color: theme.muted }}>No matches</div>
              ) : (
                searchResults.map(r => (
                  <button
                    key={r.id}
                    onClick={() => {
                      selectById(r.id);
                      setSearchQuery('');
                    }}
                    style={{
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      padding: '5px 10px',
                      border: 'none',
                      background: 'transparent',
                      color: theme.text,
                      fontSize: 12,
                      cursor: 'pointer',
                    }}
                  >
                    {r.id}
                  </button>
                ))
              )}
            </div>
          )}
        </div>

        <select
          value={layerFilter ?? ''}
          onChange={e => setLayerFilter(e.target.value === '' ? null : Number(e.target.value))}
          style={selectStyle}
          title="Filter by layer"
        >
          <option value="">All layers</option>
          {layers.map(l => (
            <option key={l} value={l}>
              Layer {l}
            </option>
          ))}
        </select>

        {heads.length > 0 && (
          <select
            value={headFilter ?? ''}
            onChange={e => setHeadFilter(e.target.value === '' ? null : Number(e.target.value))}
            style={selectStyle}
            title="Filter by head"
          >
            <option value="">All heads</option>
            {heads.map(h => (
              <option key={h} value={h}>
                Head {h}
              </option>
            ))}
          </select>
        )}

        <label
          style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: theme.muted }}
          title="Hide neurons with |activation| below this fraction of the max"
        >
          Threshold
          <input
            type="range"
            min={0}
            max={0.5}
            step={0.005}
            value={threshold}
            onChange={e => setThreshold(parseFloat(e.target.value))}
            style={{ width: 80 }}
          />
          {threshold.toFixed(2)}
        </label>

        <div style={{ display: 'flex', gap: 4 }}>
          <button
            onClick={() => setInteractionMode('pan')}
            title="Pan (drag)"
            style={interactionMode === 'pan' ? btnActive : btn}
          >
            <Hand size={13} />
          </button>
          <button
            onClick={() => setInteractionMode('select')}
            title="Box select (shift+click to add)"
            style={interactionMode === 'select' ? btnActive : btn}
          >
            <BoxSelect size={13} />
          </button>
        </div>

        <div style={{ display: 'flex', gap: 4 }}>
          <button onClick={() => zoomAt(1.3)} title="Zoom in (+)" style={btn}>
            <ZoomIn size={13} />
          </button>
          <button onClick={() => zoomAt(1 / 1.3)} title="Zoom out (-)" style={btn}>
            <ZoomOut size={13} />
          </button>
          <button onClick={fitView} title="Fit view (F)" style={btn}>
            <Maximize size={13} />
          </button>
          <button onClick={() => exportPNG(theme)} title="Export PNG" style={btn}>
            <Download size={13} />
          </button>
        </div>

        <span style={{ fontSize: 11, color: theme.muted, fontVariantNumeric: 'tabular-nums' }}>
          {Math.round(viewport.k * 100)}% · {stats.total} neurons · {stats.active} active · {stats.clusters} clusters
        </span>
      </div>

      <div
        style={{
          position: 'absolute',
          top: 42,
          left: 8,
          background: theme.cardBg,
          border: `1px solid ${theme.border}`,
          borderRadius: 8,
          padding: '6px 10px',
          fontSize: 11,
          color: theme.text,
          boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
          zIndex: 5,
          pointerEvents: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: theme.green, display: 'inline-block' }} />
          Green = more contribution
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: theme.red, display: 'inline-block' }} />
          Red = less contribution
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: theme.gray, display: 'inline-block' }} />
          Gray = neutral
        </div>
      </div>

      <canvas
        ref={minimapRef}
        style={{
          position: 'absolute',
          right: 8,
          bottom: 8,
          width: MINIMAP_W,
          height: MINIMAP_H,
          border: `1px solid ${theme.border}`,
          borderRadius: 6,
          background: theme.bg,
          zIndex: 5,
          pointerEvents: 'none',
        }}
      />

      <div
        ref={tooltipRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          display: 'none',
          maxWidth: 240,
          padding: '6px 9px',
          borderRadius: 6,
          background: theme.tooltipBg,
          color: theme.text,
          fontSize: 11,
          lineHeight: 1.4,
          pointerEvents: 'none',
          zIndex: 7,
          boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
          whiteSpace: 'nowrap',
        }}
      >
        {hoveredPoint && (
          <>
            <div style={{ fontWeight: 600 }}>{hoveredPoint.id}</div>
            <div>
              Layer {hoveredPoint.layer}
              {hoveredPoint.head !== undefined ? ` · Head ${hoveredPoint.head}` : ''}
            </div>
            <div style={{ color: hoveredPoint.activation >= 0 ? theme.green : theme.red }}>
              Activation {hoveredPoint.activation.toFixed(4)}
            </div>
            {hoveredPoint.topToken ? (
              <div style={{ color: theme.accent }}>Activates on {hoveredPoint.topToken.trim()}</div>
            ) : null}
          </>
        )}
      </div>

      {primaryPoint && (
        <NeuronDetailPanel
          point={primaryPoint}
          maxAbsAct={maxAbsAct}
          selectedCount={selectedIndices.size}
          theme={theme}
          connected={connected}
          onClose={clearSelection}
          onSelectPoint={selectById}
        />
      )}

      {loading && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            background: theme.bg,
            zIndex: 12,
          }}
        >
          <Loader2 size={22} style={{ animation: 'nuspin 0.9s linear infinite', color: theme.accent }} />
          <span style={{ fontSize: 12, color: theme.muted }}>Loading neurons…</span>
          <style>{'@keyframes nuspin { to { transform: rotate(360deg); } }'}</style>
        </div>
      )}

      {!loading && points.length === 0 && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 13,
            color: theme.muted,
            zIndex: 11,
            background: theme.bg,
          }}
        >
          No neuron data available
        </div>
      )}
    </div>
  );
}

export default NeuronUMAP;
