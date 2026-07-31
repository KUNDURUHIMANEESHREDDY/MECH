import React, { useState } from 'react';
import { Rewind } from 'lucide-react';

export default function ResearchReplayPanel() {
  const [timelineStep, setTimelineStep] = useState(2);
  const events = [
    { step: 1, title: 'Campaign Started', time: 'Week 1' },
    { step: 2, title: 'Circuit Discovered', time: 'Week 2' },
    { step: 3, title: 'Validation Completed', time: 'Week 3' },
    { step: 4, title: 'Paper Published', time: 'Week 4' }
  ];

  return (
    <div className="panel research-replay-panel" data-testid="research-replay-panel">
      <div className="panel-header">
        <h3><Rewind size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Campaign Time-Travel Research Replay</h3>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <input
          type="range"
          min="1"
          max={events.length}
          value={timelineStep}
          onChange={(e) => setTimelineStep(Number(e.target.value))}
          style={{ width: '100%', marginBottom: '12px' }}
        />
        <div style={{ padding: '12px', background: '#1e293b', borderRadius: '6px' }}>
          <h4 style={{ margin: '0 0 4px 0', color: '#38bdf8' }}>{events[timelineStep - 1].title}</h4>
          <span style={{ fontSize: '11px', color: '#94a3b8' }}>Timestamp: {events[timelineStep - 1].time}</span>
        </div>
      </div>
    </div>
  );
}
