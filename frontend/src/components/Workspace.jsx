import React, { useEffect, useState } from 'react';

export default function Workspace({ api }) {
  const [pingResult, setPingResult] = useState(null);
  const [info, setInfo] = useState(null);
  const [busy, setBusy] = useState(false);
  const [a, setA] = useState(2);
  const [b, setB] = useState(40);
  const [sum, setSum] = useState(null);

  useEffect(() => {
    if (!api) return;
    api.pythonPing().then(setPingResult).catch(() => setPingResult({ ok: false }));
  }, [api]);

  const handleInfo = async () => {
    setBusy(true);
    try {
      const result = await api.pythonCall('info', {});
      setInfo(result);
    } catch (err) {
      setInfo({ error: err.message });
    } finally {
      setBusy(false);
    }
  };

  const handleAdd = async () => {
    setBusy(true);
    try {
      const result = await api.pythonCall('add', { a: Number(a), b: Number(b) });
      setSum(result?.sum);
    } catch (err) {
      setSum(`error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="card">
        <h2>Workspace</h2>
        <p className="hint">A small dashboard to confirm the Electron shell, IPC and Python sidecar are wired together.</p>
        <div className="row">
          <label>Python ping</label>
          <div data-testid="ping-result">
            {pingResult?.ok ? <span className="tag" style={{ color: 'var(--success)' }}>OK</span>
                            : <span className="tag" style={{ color: 'var(--danger)' }}>Down</span>}
          </div>
        </div>
        <div className="toolbar">
          <button onClick={handleInfo} disabled={busy}>Get Python info</button>
          <button className="secondary" onClick={() => setPingResult(null)}>Clear</button>
        </div>
        {info && (
          <pre className="log-view" data-testid="python-info">
            {JSON.stringify(info, null, 2)}
          </pre>
        )}
      </div>

      <div className="card">
        <h2>Quick compute</h2>
        <p className="hint">Round-trips through the Python sidecar via IPC.</p>
        <div className="row">
          <label>a</label>
          <input type="number" value={a} onChange={(e) => setA(e.target.value)} />
        </div>
        <div className="row">
          <label>b</label>
          <input type="number" value={b} onChange={(e) => setB(e.target.value)} />
        </div>
        <div className="toolbar">
          <button onClick={handleAdd} disabled={busy}>Add via Python</button>
        </div>
        {sum !== null && (
          <div className="row">
            <label>Result</label>
            <div data-testid="add-result"><strong>{String(sum)}</strong></div>
          </div>
        )}
      </div>
    </div>
  );
}
