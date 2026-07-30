import React, { useState, useEffect } from 'react';
import { eventBus } from '../../utils/eventBus';

export default function TimelinePanel() {
  const [events, setEvents] = useState([
    { id: 1, ts: new Date().toLocaleTimeString(), type: 'system', label: 'Neural Debugger initialized' },
    { id: 2, ts: new Date().toLocaleTimeString(), type: 'runtime', label: 'Python sidecar process connected' }
  ]);

  useEffect(() => {
    const unsub = eventBus.on('timeline:event', (evt) => {
      setEvents((prev) => [
        {
          id: Date.now() + Math.random(),
          ts: new Date().toLocaleTimeString(),
          type: evt.type || 'info',
          label: evt.label || JSON.stringify(evt)
        },
        ...prev.slice(0, 49)
      ]);
    });
    return unsub;
  }, []);

  return (
    <div className="panel-content timeline-panel" data-testid="timeline-panel">
      <h4>Execution Timeline & Telemetry</h4>
      <p className="hint">Real-time event stream dispatched via EventBus.</p>

      <div className="timeline-stream">
        {events.map((e) => (
          <div key={e.id} className={`timeline-item type-${e.type}`}>
            <span className="timeline-ts">{e.ts}</span>
            <span className="timeline-badge">{e.type}</span>
            <span className="timeline-label">{e.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
