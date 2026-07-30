import React, { useEffect, useState } from 'react';

export default function Projects({ api }) {
  const [projects, setProjects] = useState([]);
  const [name, setName] = useState('');
  const [path, setPath] = useState('');

  const refresh = () => {
    if (!api) return;
    api.listProjects().then(setProjects).catch(() => undefined);
  };

  useEffect(refresh, [api]);

  const handleAdd = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    await api.addProject({
      id: `prj_${Date.now()}`,
      name: name.trim(),
      path: path.trim()
    });
    setName('');
    setPath('');
    refresh();
  };

  const handleRemove = async (id) => {
    await api.removeProject(id);
    refresh();
  };

  return (
    <div>
      <div className="card">
        <h2>Projects</h2>
        <p className="hint">Workspace projects are stored in the local SQLite database.</p>
        <form onSubmit={handleAdd}>
          <div className="row">
            <label>Name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="My project" />
          </div>
          <div className="row">
            <label>Path</label>
            <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="C:\\projects\\example" />
          </div>
          <div className="toolbar">
            <button type="submit">Add project</button>
          </div>
        </form>
      </div>

      <div className="card">
        <h2>Saved projects</h2>
        {projects.length === 0 ? (
          <ul className="list"><li className="empty">No projects yet.</li></ul>
        ) : (
          <ul className="list" data-testid="projects-list">
            {projects.map((p) => (
              <li key={p.id}>
                <div>
                  <div><strong>{p.name}</strong></div>
                  <div className="hint">{p.path || '—'}</div>
                </div>
                <button className="danger" onClick={() => handleRemove(p.id)}>Remove</button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
