import React, { useEffect, useRef } from 'react';
import {
  SocietyEvent,
  getSocietyRun,
  startSocietyRun,
  streamSocietyRun,
} from '../services/societyService';
import { societyStore, useSocietyStore } from '../store/useSocietyStore';

const AGENT_ORDER = ['planner', 'executor', 'inspector', 'discoverer', 'critic', 'scribe'];

function stepColor(status: string): string {
  if (status === 'completed' || status === 'ok' || status === 'loaded') return 'var(--success)';
  if (status === 'running') return 'var(--accent)';
  if (status === 'error' || status === 'unavailable' || status === 'failed') return 'var(--danger)';
  return 'var(--text-muted)';
}

function eventLabel(ev: SocietyEvent): string {
  const p = ev.payload ?? {};
  switch (ev.event_type) {
    case 'ResearchStarted':
      return `Goal: ${String(p.goal ?? '').slice(0, 80)}`;
    case 'ExperimentQueued':
      if (Array.isArray(p.plan)) return `Plan queued: ${(p.plan as string[]).join(' → ')}`;
      if (p.replan) return `Replan #${String(p.replan)} after: ${String(p.failed ?? '')}`;
      return 'Experiment queued';
    case 'DiscoveryCreated':
      return `Discovery ${String(p.discovery_id ?? '')}`;
    case 'HypothesisRejected':
      return `Stage failed: ${String(p.node ?? '')} — ${String(p.reason ?? '').slice(0, 100)}`;
    case 'CircuitValidated':
      return `Validated: ${String((p.successful as string[] ?? []).join(', ')) || '—'}`;
    case 'PublicationGenerated':
      return `Report ${String(p.experiment_id ?? '')} published`;
    case 'ResearchFinished':
      return `Finished (${String(p.steps_completed ?? '')} steps)`;
    default:
      return ev.event_type;
  }
}

// Module-level guard: at most one live subscription per run, even if two
// SocietyPanel instances mount briefly during a branch transition.
let attachedRunId: string | null = null;

export const SocietyPanel: React.FC = () => {
  const [state, setState] = useSocietyStore();
  const { goal, phase, runId, events, steps, summary, reportMd, gate, error } = state;
  const unsubRef = useRef<(() => void) | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopStreams = () => {
    unsubRef.current?.();
    unsubRef.current = null;
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (attachedRunId !== null) attachedRunId = null;
  };

  const applyDone = (result: Record<string, any>) => {
    stopStreams();
    const trace = Array.isArray(result.trace) ? result.trace : [];
    setState({ steps: trace });
    const pub = (result.publication ?? {}) as Record<string, any>;
    setState({
      summary: `${pub.steps_completed ?? '?/?'} steps · ${pub.experiment_id ?? 'no-id'} · ${result.status ?? ''}`,
    });
    const md = (pub.report as Record<string, any> | undefined)?.markdown;
    setState({ reportMd: typeof md === 'string' ? md.slice(0, 1200) : '' });
    setState({ gate: (pub.gate ?? null) as Record<string, any> | null });
    setState({ phase: result.status === 'completed' ? 'done' : 'failed' });
  };

  const startPolling = (id: string) => {
    if (pollRef.current !== null) return;
    pollRef.current = window.setInterval(async () => {
      try {
        const st = await getSocietyRun(id);
        setState({ events: st.events ?? [] });
        if (st.status !== 'running') {
          applyDone((st.result ?? {}) as Record<string, any>);
        }
      } catch (e) {
        setState({ error: `Polling failed: ${String(e)}` });
      }
    }, 2000);
  };

  const attach = (id: string) => {
    if (attachedRunId === id) return;
    attachedRunId = id;
    unsubRef.current = streamSocietyRun(id, {
      onEvent: ev => setState(prev => ({ events: [...prev.events, ev] })),
      onDone: result => applyDone(result),
      onError: () => startPolling(id),
    });
  };

  // Reattach on (re)mount: the backend replays missed events, so moving the
  // panel between the pre-inference and dock branches loses nothing.
  useEffect(() => {
    const s = societyStore.getState();
    if (s.phase === 'running' && s.runId) attach(s.runId);
    return stopStreams;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRun = async () => {
    const g = goal.trim();
    if (!g || societyStore.getState().phase === 'running') return;
    stopStreams();
    setState({ events: [], steps: [], summary: '', reportMd: '', gate: null, error: '', phase: 'running' });
    try {
      const started = await startSocietyRun(g);
      setState({ runId: started.runId });
      attach(started.runId);
    } catch (e) {
      setState({ error: String(e), phase: 'failed' });
    }
  };

  const planned = steps.length > 0 ? steps : AGENT_ORDER.map(a => ({ node: a, agent: a, status: 'pending' }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: 12 }}>
      <div style={{ display: 'flex', gap: 6 }}>
        <input
          value={goal}
          onChange={e => setState({ goal: e.target.value })}
          onKeyDown={e => {
            if (e.key === 'Enter') void handleRun();
          }}
          placeholder="Research goal, e.g. Reproduce IOI on gpt2-small…"
          className="input-text"
          style={{ flex: 1 }}
          disabled={phase === 'running'}
        />
        <button onClick={() => void handleRun()} disabled={phase === 'running' || !goal.trim()} className="btn">
          {phase === 'running' ? 'Running…' : 'Run Society'}
        </button>
      </div>

      {runId && (
        <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>
          run <span style={{ fontFamily: 'monospace' }}>{runId}</span>
          {summary && <span> · {summary}</span>}
        </div>
      )}
      {error && <div style={{ color: 'var(--danger)' }}>{error}</div>}

      <div style={{ background: 'var(--bg-elev-2)', padding: 10, borderRadius: 6 }}>
        <div style={{ fontWeight: 700, marginBottom: 6 }}>Pipeline</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {planned.map(s => (
            <div key={s.node} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span
                style={{
                  width: 9,
                  height: 9,
                  borderRadius: '50%',
                  background: stepColor(s.status),
                  flexShrink: 0,
                }}
              />
              <span style={{ fontWeight: 600 }}>{s.node}</span>
              <span style={{ color: 'var(--text-muted)' }}>{s.agent}</span>
              <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: 11 }}>
                {s.status}
                {(s.error ?? s.reason) ? ` — ${String(s.error ?? s.reason).slice(0, 60)}` : ''}
              </span>
            </div>
          ))}
        </div>
      </div>

      {reportMd && (
        <div style={{ background: 'var(--bg-elev-2)', padding: 10, borderRadius: 6 }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>Mechanistic Report</div>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: 11, color: 'var(--text-dim)' }}>
            {reportMd}
          </pre>
          {(() => {
            const patch = steps.find(s => s.node === 'patch' && s.status === 'ok');
            if (!patch || patch.layer === undefined || patch.head === undefined) return null;
            const href = `http://localhost:8000/api/figures/attention?prompt=${encodeURIComponent(goal)}&layer=${patch.layer}&head=${patch.head}`;
            return (
              <div style={{ marginTop: 8 }}>
                <a href={href} target="_blank" rel="noreferrer" className="btn" style={{ textDecoration: 'none' }}>
                  Export attention figure L{String(patch.layer)}H{String(patch.head)} (PNG)
                </a>
              </div>
            );
          })()}
        </div>
      )}

      {gate && (
        <div style={{ background: 'var(--bg-elev-2)', padding: 10, borderRadius: 6 }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Validation Gate (0.85)</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span
              style={{
                background: gate.passed ? 'var(--green)' : 'var(--red)',
                color: '#fff',
                padding: '2px 10px',
                borderRadius: 20,
                fontWeight: 700,
                fontSize: 11,
              }}
            >
              {gate.passed ? 'PASSED' : 'FAILED'}
            </span>
            <span style={{ color: 'var(--text-dim)', fontSize: 11 }}>
              fidelity {gate.value ?? '?'}% · confidence {(gate.confidence ?? 0).toFixed(2)} · validated{' '}
              {gate.validated ? 'yes' : 'no'}
            </span>
          </div>
          {!gate.passed && (
            <div style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 6 }}>
              Failing metrics are listed in the reproducibility report — minimality is unmeasured until a
              redundancy ablation exists, which caps overall fidelity by design.
            </div>
          )}
        </div>
      )}

      <div style={{ background: 'var(--bg-elev-2)', padding: 10, borderRadius: 6 }}>
        <div style={{ fontWeight: 700, marginBottom: 6 }}>Live Trace ({events.length})</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 220, overflowY: 'auto' }}>
          {events.length === 0 && (
            <div style={{ color: 'var(--text-muted)' }}>No events yet — run the Society to stream its trace.</div>
          )}
          {events.map((ev, i) => (
            <div key={i} style={{ display: 'flex', gap: 8 }}>
              <span style={{ color: 'var(--accent)', fontWeight: 700, flexShrink: 0 }}>{ev.event_type}</span>
              <span style={{ color: 'var(--text-dim)' }}>{eventLabel(ev)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
