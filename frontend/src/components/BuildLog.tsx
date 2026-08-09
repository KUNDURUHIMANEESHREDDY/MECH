import React, { useEffect, useMemo, useState } from 'react';
import { Hammer, RefreshCw, Copy, Play, Trash2 } from 'lucide-react';
import { colors } from '../design/tokens/colors';

interface LogEntry {
  timestamp: string;
  level: 'info' | 'warn' | 'error';
  message: string;
}

const levelColor = (level: LogEntry['level']) => (level === 'error' ? colors.dangerText : level === 'warn' ? colors.warningText : colors.bodyMuted);

export const BuildLog: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<'all' | 'info' | 'warn' | 'error'>('all');
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [running, setRunning] = useState(false);

  const load = async (silent = false) => {
    try {
      const entries = await window.appApi.getBuildLogs();
      setLogs(entries.slice(-100).map((e) => ({ timestamp: e.ts, level: e.level, message: e.message })));
      setStatus('ready');
    } catch {
      if (!silent) setStatus('error');
    }
  };

  useEffect(() => {
    void load();
    let unsub: (() => void) | undefined;
    try {
      unsub = window.appApi.onBuildEvent((ev) => {
        const level: LogEntry['level'] = ev.type === 'stderr' ? 'warn' : ev.type === 'error' ? 'error' : 'info';
        const message = typeof ev.data === 'string' ? ev.data : `build_finished code=${(ev.data as { code?: number }).code ?? '?'}`;
        setLogs((prev) => [...prev, { timestamp: ev.ts, level, message }].slice(-100));
        if (ev.type === 'close') setRunning(false);
      });
    } catch {
      void 0;
    }
    return () => {
      if (unsub) unsub();
    };
  }, []);

  const runBuild = async () => {
    setRunning(true);
    try {
      const res = await window.appApi.startBuild({ target: 'renderer' });
      if (res && res.ok === false) setRunning(false);
    } catch {
      setRunning(false);
      setStatus('error');
    }
  };

  const clear = async () => {
    try {
      await window.appApi.clearBuildLogs();
      setLogs([]);
    } catch {
      void 0;
    }
  };

  const visible = useMemo(() => logs.filter((l) => filter === 'all' || l.level === filter), [logs, filter]);
  const counts = useMemo(() => ({ info: logs.filter((l) => l.level === 'info').length, warn: logs.filter((l) => l.level === 'warn').length, error: logs.filter((l) => l.level === 'error').length }), [logs]);

  const copyLogs = () => {
    const text = logs.map((l) => `[${l.timestamp}] ${l.level.toUpperCase()} ${l.message}`).join('\n');
    void navigator.clipboard?.writeText(text);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Hammer size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Build Log</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{logs.length} entries · {counts.error} errors</span>
        <button onClick={() => void runBuild()} disabled={running} style={{ background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '5px 12px', fontSize: 11, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 5, opacity: running ? 0.6 : 1 }}>
          <Play size={11} /> {running ? 'Building…' : 'Run build'}
        </button>
        <button onClick={() => void clear()} style={{ background: 'none', border: 'none', cursor: 'pointer', color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
          <Trash2 size={12} />
        </button>
        <button onClick={() => void load()} style={{ background: 'none', border: 'none', cursor: 'pointer', color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
          <RefreshCw size={12} />
        </button>
        <button onClick={copyLogs} style={{ background: 'none', border: 'none', cursor: 'pointer', color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
          <Copy size={12} />
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
            {f} {f === 'all' ? logs.length : counts[f]}
          </button>
        ))}
      </div>

      {status === 'error' && <div style={{ fontSize: 12, color: colors.dangerText }}>Build bridge unavailable — running outside the Electron shell?</div>}

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 8, maxHeight: '60vh', overflowY: 'auto', fontFamily: 'monospace', fontSize: 11 }}>
        {visible.length === 0 && <div style={{ color: colors.bodyMuted, padding: 8 }}>No build output yet. Press Run build to compile the renderer.</div>}
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