import React, { useEffect, useMemo, useState } from 'react';
import { ScrollText, RefreshCw, ThumbsUp } from 'lucide-react';
import { colors } from '../design/tokens/colors';

interface LogEntry {
  timestamp: string;
  level: 'info' | 'warn' | 'error';
  message: string;
}

const levelColor = (level: LogEntry['level']) => (level === 'error' ? colors.dangerText : level === 'warn' ? colors.warningText : colors.bodyMuted);

const toLevel = (raw: string): LogEntry['level'] => (raw === 'warn' || raw === 'error' ? raw : 'info');

export const Logging: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<'all' | 'info' | 'warn' | 'error'>('all');

  const load = async () => {
    try {
      const entries = await window.appApi.getAppLogs();
      setLogs(entries.slice(-150).map((e) => ({ timestamp: e.ts, level: toLevel(e.level), message: e.event })));
    } catch {
      void 0;
    }
  };

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 4000);
    return () => window.clearInterval(timer);
  }, []);

  const counts = useMemo(() => ({ info: logs.filter((l) => l.level === 'info').length, warn: logs.filter((l) => l.level === 'warn').length, error: logs.filter((l) => l.level === 'error').length }), [logs]);
  const visible = useMemo(() => logs.filter((l) => filter === 'all' || l.level === filter), [logs, filter]);
  const total = counts.info + counts.warn + counts.error;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <ScrollText size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Execution Logs</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>live · {total} total</span>
        <button onClick={() => void load()} style={{ background: 'none', border: 'none', cursor: 'pointer', color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
          <RefreshCw size={12} />
        </button>
      </div>

      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {(['all', 'info', 'warn', 'error'] as const).map((f) => (
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
            {f === 'all' ? `all ${total}` : `${f} ${counts[f]}`}
          </button>
        ))}
        {counts.error === 0 && (
          <span style={{ fontSize: 11, color: colors.successText, display: 'inline-flex', alignItems: 'center', gap: 4, background: colors.successSoft, borderRadius: 12, padding: '3px 10px' }}>
            <ThumbsUp size={11} /> no errors
          </span>
        )}
      </div>

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 8, maxHeight: '60vh', overflowY: 'auto', fontFamily: 'monospace', fontSize: 11 }}>
        {visible.length === 0 && <div style={{ color: colors.bodyMuted, padding: 8 }}>No log entries yet.</div>}
        {visible.slice().reverse().map((l, i) => (
          <div key={`${l.timestamp}_${i}`} style={{ display: 'flex', gap: 8, padding: '2px 4px', borderBottom: i < visible.length - 1 ? `1px solid ${colors.dividerSoft}` : 'none' }}>
            <span style={{ color: colors.bodyMuted, whiteSpace: 'nowrap' }}>{l.timestamp.slice(11, 19)}</span>
            <span style={{ color: levelColor(l.level), fontWeight: 700, whiteSpace: 'nowrap' }}>{l.level.toUpperCase()}</span>
            <span style={{ color: colors.ink, overflowWrap: 'anywhere' }}>{l.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
};