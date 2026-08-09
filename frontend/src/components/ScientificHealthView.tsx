import React, { useEffect, useState } from 'react';
import { HeartPulse, RefreshCw } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { useModel } from '../shared/hooks/useModel';
import { useWorkspaceStore } from '../shared/stores/workspace';

interface Check {
  label: string;
  ok: boolean;
  detail: string;
}

export const ScientificHealthView: React.FC = () => {
  const { state: model } = useModel();
  const { console: consoleLogs, timeline } = useWorkspaceStore();
  const [sidecar, setSidecar] = useState<boolean | null>(null);
  const [lastCheck, setLastCheck] = useState<string | null>(null);

  const refresh = async () => {
    try {
      const api = window.desktopApi;
      if (!api) {
        setSidecar(false);
        setLastCheck(new Date().toLocaleTimeString());
        return;
      }
      const pong = await Promise.race([
        api.ping().then(() => true),
        new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 3000)),
      ]);
      setSidecar(pong);
    } catch {
      setSidecar(false);
    }
    setLastCheck(new Date().toLocaleTimeString());
  };

  useEffect(() => {
    void refresh();
  }, []);

  const errors = consoleLogs.filter((c) => c.level === 'error').length;
  const warns = consoleLogs.filter((c) => c.level === 'warn').length;

  const checks: Check[] = [
    { label: 'Model loaded', ok: model.loaded, detail: model.modelInfo?.model_name ?? (model.loading ? 'loading…' : 'none') },
    { label: 'Python sidecar reachable', ok: sidecar === true, detail: sidecar === null ? 'checking…' : sidecar ? 'ping ok' : 'unreachable' },
    { label: 'Inference available', ok: model.loaded && !model.running, detail: model.running ? 'running' : model.loaded ? 'idle' : 'load a model' },
    { label: 'Console errors', ok: errors === 0, detail: `${errors} error${errors === 1 ? '' : 's'}` },
    { label: 'Console warnings', ok: warns === 0, detail: `${warns} warning${warns === 1 ? '' : 's'}` },
    { label: 'Activity recorded', ok: timeline.length > 0, detail: `${timeline.length} events` },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <HeartPulse size={16} color={colors.success} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Scientific Health</span>
        {lastCheck && <span style={{ fontSize: 11, color: colors.bodyMuted }}>last checked {lastCheck}</span>}
        <button onClick={() => void refresh()} style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
          <RefreshCw size={12} /> Re-check
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {checks.map((c) => (
          <div key={c.label} style={{ display: 'flex', alignItems: 'center', gap: 10, border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: '10px 12px' }}>
            <span style={{ width: 10, height: 10, borderRadius: 5, background: c.ok ? colors.success : colors.dangerText, flexShrink: 0 }} />
            <span style={{ color: colors.ink, fontWeight: 600, width: 190 }}>{c.label}</span>
            <span style={{ color: c.ok ? colors.body : colors.dangerText, fontSize: 12 }}>{c.detail}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
