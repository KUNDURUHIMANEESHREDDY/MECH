import React, { useEffect, useState } from 'react';
import { FolderOpen, FolderPlus, RefreshCw, Clock, Trash2 } from 'lucide-react';
import { colors } from '../design/tokens/colors';

interface Project {
  id: string;
  path: string;
  name: string;
  updatedAt: string;
}

const formatDate = (iso: string) => {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
};

export const Projects: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [path, setPath] = useState('');
  const [name, setName] = useState('');

  const refresh = async () => {
    setStatus('loading');
    try {
      const list = await window.appApi.listProjects();
      setProjects(list.slice(0, 20).map((p) => ({ id: p.id, path: p.path, name: p.name, updatedAt: p.updatedAt || p.createdAt })));
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
      await window.appApi.addProject({ id: `proj_${Date.now()}`, path: path.trim(), name: name.trim() || undefined });
      setPath('');
      setName('');
      void refresh();
    } catch {
      void 0;
    }
  };

  const remove = async (id: string) => {
    try {
      await window.appApi.removeProject(id);
      void refresh();
    } catch {
      void 0;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <FolderOpen size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Research Projects</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{projects.length} projects</span>
        <button onClick={() => void refresh()} style={{ background: 'none', border: 'none', cursor: 'pointer', color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11 }}>
          <RefreshCw size={12} />
        </button>
      </div>

      {status === 'error' && <div style={{ fontSize: 12, color: colors.dangerText }}>Project bridge unavailable — running outside the Electron shell?</div>}
      {status === 'loading' && <div style={{ fontSize: 12, color: colors.bodyMuted }}>Loading projects…</div>}

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder="Project path"
          style={{ flex: 1, minWidth: 180, padding: 6, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontSize: 12, background: colors.canvas, color: colors.ink }}
        />
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Name (optional)"
          style={{ flex: 1, minWidth: 120, padding: 6, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontSize: 12, background: colors.canvas, color: colors.ink }}
        />
        <button onClick={() => void add()} disabled={!path.trim()} style={{ background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '6px 12px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: !path.trim() ? 0.5 : 1 }}>
          <FolderPlus size={13} /> Add
        </button>
      </div>

      {projects.length === 0 && status === 'ready' && (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          No projects recorded yet. Add a path above.
        </div>
      )}

      {projects.map((p) => (
        <div key={p.id} style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: '10px 12px', display: 'flex', alignItems: 'center', gap: 10 }}>
          <FolderOpen size={14} color={colors.primary} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontWeight: 700, color: colors.ink }}>{p.name || '(untitled)'}</div>
            <div style={{ fontSize: 11, color: colors.bodyMuted, fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.path}</div>
          </div>
          <span style={{ fontSize: 11, color: colors.bodyMuted, display: 'inline-flex', alignItems: 'center', gap: 4, whiteSpace: 'nowrap' }}>
            <Clock size={11} /> {formatDate(p.updatedAt)}
          </span>
          <button onClick={() => void remove(p.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: colors.dangerText, display: 'inline-flex' }}>
            <Trash2 size={13} />
          </button>
        </div>
      ))}
    </div>
  );
};