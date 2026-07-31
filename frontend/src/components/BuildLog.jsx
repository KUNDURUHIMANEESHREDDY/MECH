import React, { useEffect, useRef, useState } from 'react';

export default function BuildLog({ api }) {
  const [logs, setLogs] = useState([]);
  const [running, setRunning] = useState(false);
  const [target, setTarget] = useState('renderer');
  const [exitCode, setExitCode] = useState(null);
  const viewRef = useRef(null);

  const refresh = async () => {
    if (!api) return;
    const items = await api.getBuildLogs();
    setLogs(items);
  };

  useEffect(() => { refresh(); }, [api]);

  useEffect(() => {
    if (!api) return;
    const off = api.onBuildEvent((evt) => {
      if (evt.type === 'close') {
        setRunning(false);
        setExitCode(evt.data?.code ?? null);
      } else if (evt.type === 'error') {
        setLogs((prev) => [...prev, { ts: evt.ts, level: 'error', message: String(evt.data) }]);
      }
      // stdout/stderr are appended by the main process into the log buffer;
      // refresh to pull them.
      refresh();
    });
    return off;
  }, [api]);

  // Auto-scroll to bottom on new entries
  useEffect(() => {
    if (viewRef.current) {
      viewRef.current.scrollTop = viewRef.current.scrollHeight;
    }
  }, [logs]);

  const start = async () => {
    setLogs([]);
    setExitCode(null);
    setRunning(true);
    await api.startBuild({ target });
  };

  const clear = async () => {
    await api.clearBuildLogs();
    setLogs([]);
    setExitCode(null);
  };

  return (
    <div>
      <div className="card">
        <h2>Build</h2>
        <p className="hint">Run a build target and watch its output stream live.</p>
        <div className="row">
          <label>Target</label>
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            <option value="renderer">Renderer (vite build)</option>
            <option value="python">Python (pip --version)</option>
          </select>
        </div>
        <div className="toolbar">
          <button onClick={start} disabled={running} data-testid="build-start">
            {running ? 'Running…' : 'Start build'}
          </button>
          <button className="secondary" onClick={clear} disabled={running}>Clear logs</button>
          {exitCode !== null && (
            <span className="tag" style={{ color: exitCode === 0 ? 'var(--success)' : 'var(--danger)' }}>
              exit {exitCode}
            </span>
          )}
        </div>
        <div className="log-view" ref={viewRef} data-testid="build-log">
          {logs.length === 0 ? (
            <div style={{ color: 'var(--text-dim)' }}>No output yet. Press “Start build”.</div>
          ) : (
            logs.map((l, i) => (
              <div key={i} className={`log-line ${l.level}`}>
                <span className="ts">{l.ts}</span>
                <span className="msg">{l.message}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

