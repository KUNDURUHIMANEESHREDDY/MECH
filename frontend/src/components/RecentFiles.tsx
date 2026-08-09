import React, { useEffect, useState } from 'react';
import { FileText, FilePlus2, RefreshCw, Clock } from 'lucide-react';
import { colors } from '../design/tokens/colors';

interface RecentFile {
  id: string;
  path: string;
  label: string;
  openedAt: string;
}

const formatDate = (iso: string) => {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
};

export const RecentFiles: React.FC = () => {
  const [files, setFiles] = useState<RecentFile[]>([]);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [path, setPath] = useState('');

  const refresh = async () => {
    setStatus('loading');
    try {
      const list = await window.appApi.listRecentFiles();
      setFiles(list.slice(0, 20).map((f) => ({ id: f.id, path: f.path, label: f.label, openedAt: f.openedAt })));
      setStatus('ready');
    } catch {
      setStatus('error');
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const add = async () => {
    if (!path.trim()) return;
    try {
      await window.appApi.addRecentFile({ path: path.trim() });
      setPath('');
      void refresh();
    } catch {
      void 0;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <FileText size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Recent Files</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{files.length} recent</span>
        <button onClick={() => void refresh()} style={{ background: 'none', border: 'none', cursor: 'pointer', color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
          <RefreshCw size={12} />
        </button>
      </div>

      {status === 'error' && <div style={{ fontSize: 12, color: colors.dangerText }}>Recent-files bridge unavailable — running outside the Electron shell?</div>}
      {status === 'loading' && <div style={{ fontSize: 12, color: colors.bodyMuted }}>Loading…</div>}

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', gap: 8 }}>
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder="File path to track"
          style={{ flex: 1, padding: 6, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontSize: 12, background: colors.canvas, color: colors.ink }}
        />
        <button onClick={() => void add()} disabled={!path.trim()} style={{ background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '6px 12px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: !path.trim() ? 0.5 : 1 }}>
          <FilePlus2 size={13} /> Track
        </button>
      </div>

      {files.length === 0 && status === 'ready' && (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          No recent files recorded yet.
        </div>
      )}

      {files.map((f) => (
        <div key={f.id} style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: '10px 12px', display: 'flex', alignItems: 'center', gap: 10 }}>
          <FileText size={14} color={colors.primary} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 12, color: colors.ink, fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.label || f.path}</div>
            {f.label && f.label !== f.path && <div style={{ fontSize: 11, color: colors.bodyMuted }}>{f.path}</div>}
          </div>
          <span style={{ fontSize: 11, color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, whiteSpace: 'nowrap' }}>
            <Clock size={11} /> {formatDate(f.openedAt)}
          </span>
        </div>
      ))}
    </div>
  );
};