import React, { useState, useEffect } from 'react';
import { eventBus } from '../utils/eventBus';

export default function SessionTimelineView() {
  const [events, setEvents] = useState([
    { id: 1, type: 'start', label: 'Started inference on GPT-2 Small ("The capital of France is")', time: '20:40:00' },
    { id: 2, type: 'breakpoint', label: 'Hit Breakpoint at Layer 8', time: '20:40:02' },
    { id: 3, type: 'patch', label: 'Applied Replacement Patch on L8_N402 (+4.12)', time: '20:40:05' },
    { id: 4, type: 'continue', label: 'Resumed inference from Layer 8 to Prediction', time: '20:40:08' },
    { id: 5, type: 'observe', label: 'Observed prediction change to " Paris" (p=0.82)', time: '20:40:10' },
  ]);

  useEffect(() => {
    const unsub = eventBus.on('timeline:event', (ev) => {
      setEvents((prev) => [...prev, { id: Date.now(), type: ev.type, label: ev.label, time: new Date().toLocaleTimeString() }]);
    });
    return unsub;
  }, []);

  return (
    <div className="card session-timeline-card" data-testid="session-timeline-view">
      <h3>Interactive Session Timeline</h3>
      <p className="hint">Live trace of execution events: Run ➔ Breakpoint ➔ Patch ➔ Observe ➔ Export.</p>

      <div className="timeline-events" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {events.map((ev) => (
          <div key={ev.id} className="timeline-event-row" style={{ display: 'flex', gap: '10px', alignItems: 'center', fontSize: '12px', padding: '6px 10px', background: 'var(--bg)', borderRadius: '4px' }}>
            <span className="kbd-shortcut">{ev.time}</span>
            <span style={{ fontWeight: 'bold', color: ev.type === 'patch' ? 'var(--warning)' : 'var(--accent)' }}>[{ev.type.toUpperCase()}]</span>
            <span>{ev.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
