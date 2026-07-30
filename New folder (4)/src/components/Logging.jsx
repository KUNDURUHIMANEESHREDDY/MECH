import React, { useEffect, useState } from 'react';

export default function Logging({ api }) {
  const [entries, setEntries] = useState([]);

  const refresh = () => {
    if (!api) return;
    api.getAppLogs().then(setEntries).catch(() => undefined);
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [api]);

  return (
    <div>
      <div className="card">
        <h2>Application logs</h2>
        <p className="hint">In-memory ring buffer of app events (most recent first).</p>
        <div className="toolbar">
          <button onClick={refresh}>Refresh</button>
        </div>
        {entries.length === 0 ? (
          <ul className="list"><li className="empty">No log entries yet.</li></ul>
        ) : (
          <ul className="list" data-testid="app-logs">
            {entries.slice().reverse().map((e, i) => (
              <li key={i}>
                <div>
                  <div><strong>{e.event}</strong> <span className="tag">{e.level}</span></div>
                  <div className="hint">{e.ts}</div>
                </div>
                {e.meta && <code className="hint">{JSON.stringify(e.meta)}</code>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
