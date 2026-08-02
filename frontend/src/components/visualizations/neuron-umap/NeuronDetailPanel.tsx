import { useMemo } from 'react';
import type { CSSProperties } from 'react';
import { X } from 'lucide-react';
import { NeuronPoint, NeuronUMapTheme } from './types';

export interface ConnectedNeuron {
  point: NeuronPoint;
  dist: number;
}

export interface NeuronDetailPanelProps {
  point: NeuronPoint;
  maxAbsAct: number;
  selectedCount: number;
  theme: NeuronUMapTheme;
  connected: ConnectedNeuron[];
  onClose: () => void;
  onSelectPoint: (id: string) => void;
}

function fmt(v: number): string {
  return v.toFixed(4);
}

export function NeuronDetailPanel({
  point,
  maxAbsAct,
  selectedCount,
  theme,
  connected,
  onClose,
  onSelectPoint,
}: NeuronDetailPanelProps) {
  const importancePct = maxAbsAct > 0 ? Math.min(100, (Math.abs(point.activation) / maxAbsAct) * 100) : 0;

  const topTokens = useMemo(() => {
    const ta = point.tokenActivations;
    const tk = point.tokens;
    if (!ta) return [];
    const best: { idx: number; v: number }[] = [];
    for (let i = 0; i < ta.length; i++) {
      const av = Math.abs(ta[i]);
      if (best.length < 8) {
        best.push({ idx: i, v: ta[i] });
        best.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      } else if (av > Math.abs(best[best.length - 1].v)) {
        best[best.length - 1] = { idx: i, v: ta[i] };
        best.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      }
    }
    return best.map(b => ({
      token: tk && tk[b.idx] !== undefined ? tk[b.idx] : `[${b.idx}]`,
      v: b.v,
    }));
  }, [point]);

  const strip = useMemo(() => {
    const ta = point.tokenActivations;
    const tk = point.tokens;
    if (!ta) return [];
    const n = Math.min(ta.length, 120);
    const arr: { token: string; v: number }[] = [];
    for (let i = 0; i < n; i++) {
      arr.push({ token: tk && tk[i] !== undefined ? tk[i] : String(i), v: ta[i] });
    }
    return arr;
  }, [point]);

  const maxStrip = useMemo(() => {
    let m = 0;
    for (const s of strip) m = Math.max(m, Math.abs(s.v));
    return m || 1;
  }, [strip]);

  const ws = point.weightStats;

  const sectionTitle: CSSProperties = {
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    color: theme.muted,
    margin: '14px 0 6px',
  };

  const valueStyle: CSSProperties = {
    fontSize: 12,
    color: theme.text,
    fontVariantNumeric: 'tabular-nums',
  };

  const row: CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '3px 0',
    fontSize: 12,
  };

  const keyStyle: CSSProperties = { color: theme.muted };

  const chipStyle: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    padding: '1px 6px',
    borderRadius: 4,
    fontSize: 11,
    background: theme.bg,
    border: `1px solid ${theme.border}`,
    color: theme.text,
  };

  return (
    <div
      style={{
        position: 'absolute',
        top: 8,
        right: 8,
        bottom: 8,
        width: 300,
        background: theme.cardBg,
        border: `1px solid ${theme.border}`,
        borderRadius: 10,
        boxShadow: '0 6px 24px rgba(0,0,0,0.18)',
        zIndex: 10,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 12px',
          borderBottom: `1px solid ${theme.border}`,
        }}
      >
            <div style={{ fontSize: 13, fontWeight: 600, color: theme.text }}>
              {point.id}
              {point.topToken ? (
                <span style={{ color: theme.accent, marginLeft: 6, fontSize: 12 }}>· {point.topToken.trim()}</span>
              ) : null}
            </div>
        <button
          onClick={onClose}
          aria-label="Close neuron details"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 24,
            height: 24,
            border: 'none',
            borderRadius: 6,
            background: 'transparent',
            color: theme.muted,
            cursor: 'pointer',
          }}
        >
          <X size={14} />
        </button>
      </div>

      <div style={{ overflowY: 'auto', padding: '4px 12px 12px', flex: 1 }}>
        <div style={sectionTitle}>Identity</div>
        <div style={row}>
          <span style={keyStyle}>Layer</span>
          <span style={valueStyle}>{point.layer}</span>
        </div>
        <div style={row}>
          <span style={keyStyle}>Neuron index</span>
          <span style={valueStyle}>{point.neuronIndex}</span>
        </div>
        {point.head !== undefined && (
          <div style={row}>
            <span style={keyStyle}>Head</span>
            <span style={valueStyle}>{point.head}</span>
          </div>
        )}

        <div style={sectionTitle}>Activation &amp; importance</div>
        <div style={row}>
          <span style={keyStyle}>Activation</span>
          <span style={valueStyle}>{fmt(point.activation)}</span>
        </div>
        <div style={row}>
          <span style={keyStyle}>Importance</span>
          <span style={valueStyle}>{importancePct.toFixed(1)}% of max</span>
        </div>
        <div
          style={{
            height: 4,
            borderRadius: 2,
            background: theme.bg,
            overflow: 'hidden',
            marginTop: 2,
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${importancePct}%`,
              background: theme.accent,
              borderRadius: 2,
            }}
          />
        </div>

        {topTokens.length > 0 && (
          <>
            <div style={sectionTitle}>Top tokens</div>
            {topTokens.map((t, i) => (
              <div key={i} style={row}>
                <span style={chipStyle} title={t.token}>
                  {t.token.length > 14 ? `${t.token.slice(0, 14)}…` : t.token}
                </span>
                <span style={{ ...valueStyle, color: t.v >= 0 ? theme.green : theme.red }}>
                  {t.v >= 0 ? '+' : ''}
                  {fmt(t.v)}
                </span>
              </div>
            ))}
          </>
        )}

        {connected.length > 0 && (
          <>
            <div style={sectionTitle}>Connected neurons</div>
            {connected.map(c => (
              <button
                key={c.point.id}
                onClick={() => onSelectPoint(c.point.id)}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  width: '100%',
                  padding: '4px 6px',
                  border: 'none',
                  borderRadius: 6,
                  background: 'transparent',
                  cursor: 'pointer',
                  fontSize: 12,
                  color: theme.text,
                }}
              >
                <span>{c.point.id}</span>
                <span style={{ color: theme.muted, fontVariantNumeric: 'tabular-nums' }}>{c.dist.toFixed(4)}</span>
              </button>
            ))}
          </>
        )}

        <div style={sectionTitle}>Weight statistics</div>
        {ws ? (
          <>
            <div style={row}>
              <span style={keyStyle}>Mean</span>
              <span style={valueStyle}>{fmt(ws.mean)}</span>
            </div>
            <div style={row}>
              <span style={keyStyle}>Std dev</span>
              <span style={valueStyle}>{fmt(ws.std)}</span>
            </div>
            <div style={row}>
              <span style={keyStyle}>Min</span>
              <span style={valueStyle}>{fmt(ws.min)}</span>
            </div>
            <div style={row}>
              <span style={keyStyle}>Max</span>
              <span style={valueStyle}>{fmt(ws.max)}</span>
            </div>
          </>
        ) : (
          <div style={{ fontSize: 12, color: theme.muted }}>—</div>
        )}

        {strip.length > 0 && (
          <>
            <div style={sectionTitle}>Example activations</div>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 34 }}>
              {strip.map((s, i) => (
                <div
                  key={i}
                  title={`${s.token}: ${fmt(s.v)}`}
                  style={{
                    width: 2,
                    height: `${Math.max(4, (Math.abs(s.v) / maxStrip) * 32)}px`,
                    background: s.v >= 0 ? theme.green : s.v < 0 ? theme.red : theme.gray,
                  }}
                />
              ))}
            </div>
          </>
        )}
      </div>

      {selectedCount > 1 && (
        <div
          style={{
            padding: '6px 12px',
            borderTop: `1px solid ${theme.border}`,
            fontSize: 11,
            color: theme.muted,
          }}
        >
          {selectedCount} neurons selected — showing primary
        </div>
      )}
    </div>
  );
}
