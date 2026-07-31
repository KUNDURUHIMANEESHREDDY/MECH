import React, { useState } from 'react';
import { ChartBar } from 'lucide-react';

export default function PresentationModePanel() {
  const [slide, setSlide] = useState(1);
  const slides = [
    { title: 'Slide 1: Hypotheses & Experimental Setup', content: 'Testing L8_N402 indirect object name retrieval.' },
    { title: 'Slide 2: Live Activation Patching Demo', content: 'Zeroing L8_N402 shifts output logit from Mary to John.' },
    { title: 'Slide 3: Cross-Model Circuit Comparison', content: 'GPT-2 vs Gemma-2B IOI circuit alignment r=0.91.' }
  ];

  return (
    <div className="panel presentation-mode-panel" data-testid="presentation-mode-panel">
      <div className="panel-header">
        <h3><ChartBar size={13} style={{ verticalAlign: 'middle', marginRight: 6, color: 'var(--accent)' }} /> Presentation Mode</h3>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button className="btn" style={{ fontSize: '10px' }} onClick={() => setSlide((s) => Math.max(1, s - 1))}>Previous</button>
          <span style={{ fontSize: '11px', color: '#94a3b8', alignSelf: 'center' }}>Slide {slide} / {slides.length}</span>
          <button className="btn" style={{ fontSize: '10px' }} onClick={() => setSlide((s) => Math.min(slides.length, s + 1))}>Next</button>
        </div>
      </div>
      <div className="panel-body" style={{ padding: '12px' }}>
        <div style={{ padding: '16px', background: '#020617', borderRadius: '8px', border: '1px solid #1e293b' }}>
          <h4 style={{ margin: '0 0 6px 0', fontSize: '14px', color: '#38bdf8' }}>{slides[slide - 1].title}</h4>
          <p style={{ margin: 0, fontSize: '12px', color: '#cbd5e1' }}>{slides[slide - 1].content}</p>
        </div>
      </div>
    </div>
  );
}
