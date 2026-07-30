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
    <div className="sessions-view" data-testid="sessions-view">
      <div className="section-header">
        <h2>Session Explorer</h2>
        <p>Review past neural debugging sessions, state snapshots, and trace records.</p>
        <button className="btn btn-primary btn-sm" style={{ marginTop: '10px' }} onClick={handleCreateSession}>
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
              <div key={s.id} className="session-item">
                <div className="session-main">
                  <h4>{s.name}</h4>
                  <p className="session-meta">Model: <strong>{s.model}</strong> | Created: {new Date(s.createdAt).toLocaleString()}</p>
                  <p className="session-prompt">"{s.prompt}"</p>
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
