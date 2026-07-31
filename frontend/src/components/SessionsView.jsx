import React, { useState, useEffect } from 'react';

export default function SessionsView({ api }) {
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    if (api && api.listSessions) {
      api.listSessions().then((res) => setSessions(res || [])).catch(() => undefined);
    }
  }, [api]);

  const handleCreateSession = async () => {
    const newSession = {
      id: `sess_${Date.now()}`,
      name: `Research Session #${sessions.length + 1}`,
      model: 'GPT-2 Small',
      prompt: 'When Mary and John went to the store...',
      createdAt: new Date().toISOString(),
      metadata: { targetHeads: [9.9, 10.0] }
    };
    if (api && api.saveSession) {
      await api.saveSession(newSession);
      const updated = await api.listSessions();
      setSessions(updated || []);
    } else {
      setSessions([newSession, ...sessions]);
    }
  };

  return (
    <div>
      <div className="section-header">
        <h2>Session Explorer</h2>
        <p className="hint">Review past neural debugging sessions, state snapshots, and trace records.</p>
        <button className="btn btn-sm" style={{ marginTop: 10 }} onClick={handleCreateSession}>
          + New Research Session
        </button>
      </div>

      <div className="card">
        <h3>Recent Research Sessions</h3>
        {sessions.length === 0 ? (
          <div className="empty-state">
            <p className="hint">No sessions saved. Start a debugging session from the Debugger or click above.</p>
          </div>
        ) : (
          <div className="sessions-list">
            {sessions.map((s) => (
              <div key={s.id} className="card" style={{ marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h4 style={{ margin: 0 }}>{s.name}</h4>
                  <p className="hint" style={{ margin: '4px 0 0' }}>Model: <strong>{s.model}</strong> | Created: {new Date(s.createdAt).toLocaleString()}</p>
                  <p className="hint" style={{ margin: '2px 0 0' }}>"{s.prompt}"</p>
                </div>
                <button className="btn btn-secondary btn-sm">Load Session</button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
