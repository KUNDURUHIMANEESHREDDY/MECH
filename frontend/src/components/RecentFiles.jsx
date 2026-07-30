import React, { useEffect, useState } from 'react';

export default function RecentFiles({ api }) {
  const [files, setFiles] = useState([]);
  const [path, setPath] = useState('');

  const refresh = () => {
    if (!api) return;
    api.listRecentFiles().then(setFiles).catch(() => undefined);
  };

  useEffect(refresh, [api]);

  const handleAdd = async (e) => {
    e.preventDefault();
    if (!path.trim()) return;
    await api.addRecentFile({ path: path.trim() });
    setPath('');
    refresh();
  };

  return (
    <div>
      <div className="card">
        <h2>Recent files</h2>
        <p className="hint">Track recently opened files for quick access.</p>
        <form onSubmit={handleAdd}>
          <div className="row">
            <label>File path</label>
            <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="C:\\path\\to\\file.txt" />
          </div>
          <div className="toolbar">
            <button type="submit">Add to recent</button>
            <button type="button" className="secondary" onClick={async () => { await api.clearRecentFiles(); refresh(); }}>
              Clear list
            </button>
          </div>
        </form>
      </div>

      <div className="card">
        <h2>History</h2>
        {files.length === 0 ? (
          <ul className="list"><li className="empty">No recent files.</li></ul>
        ) : (
          <ul className="list" data-testid="recent-list">
            {files.map((f) => (
              <li key={f.id}>
                <div>
                  <div><strong>{f.label}</strong></div>
                  <div className="hint">{f.path}</div>
                </div>
                <button className="secondary" onClick={() => api.showInFolder(f.path)}>Reveal</button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
