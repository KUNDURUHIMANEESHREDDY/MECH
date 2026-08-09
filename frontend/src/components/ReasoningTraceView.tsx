import React, { useMemo, useState } from 'react';
import { Search, Puzzle, MessageSquare, AlertCircle, Info, ListFilter } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useWorkspaceStore } from '../shared/stores/workspace';

export const ReasoningTraceView: React.FC = () => {
  const { timeline, console: consoleLogs } = useWorkspaceStore();
  const [filter, setFilter] = useState<'all' | 'event' | 'console' | 'error'>('all');

  const entries = useMemo(() => {
    const rows = [
      ...timeline.map((e) => ({ ts: e.ts, kind: 'event', text: e.event })),
      ...consoleLogs.map((c) => ({ ts: c.ts, kind: c.level === 'error' ? 'error' : 'console', text: c.message })),
    ];
    rows.sort((a, b) => (a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : 0));
    return rows.slice(-80);
  }, [timeline, consoleLogs]);

  const visible = entries.filter((e) => filter === 'all' || e.kind === filter);
  const errorCount = entries.filter((e) => e.kind === 'error').length;

  const iconFor = (kind: string) => {
    const style = { width: 26, height: 26, borderRadius: 13, display: 'flex' as const, alignItems: 'center' as const, justifyContent: 'center' as const, flexShrink: 0 };
    if (kind === 'error') return <span style={{ ...style, background: colors.dangerSoft, color: colors.dangerText }}><AlertCircle size={13} /></span>;
    if (kind === 'console') return <span style={{ ...style, background: colors.infoSoft, color: colors.infoText }}><MessageSquare size={13} /></span>;
    return <span style={{ ...style, background: colors.accentSoft, color: colors.primary }}><Puzzle size={13} /></span>;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Search size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Reasoning Trace</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{visible.length} events · {errorCount} errors</span>
      </div>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
        <ListFilter size={13} color={colors.bodyMuted} />
        {(['all', 'event', 'console', 'error'] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              background: filter === f ? colors.primary : colors.canvas,
              color: filter === f ? colors.onPrimary : colors.bodyMuted,
              border: `1px solid ${filter === f ? colors.primary : colors.hairline}`,
              borderRadius: 12,
              padding: '3px 10px',
              fontSize: 11,
              cursor: 'pointer',
            }}
          >
            {f}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 8, maxHeight: '60vh', overflowY: 'auto' }}>
        {visible.length === 0 && (
          <div style={{ fontSize: 12, color: colors.bodyMuted, padding: 12 }}>No trace entries yet — run inference or open panels to record activity.</div>
        )}
        {visible.slice().reverse().map((e, i) => (
          <div key={`${e.ts}_${i}`} style={{ display: 'flex', gap: 8, alignItems: 'flex-start', padding: '5px 4px', borderBottom: i < visible.length - 1 ? `1px solid ${colors.dividerSoft}` : 'none' }}>
            {iconFor(e.kind)}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, color: colors.ink, overflowWrap: 'anywhere' }}>{e.text}</div>
              <div style={{ fontSize: 10, color: colors.bodyMuted }}>{e.ts} · {e.kind}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
