import React, { useState, useEffect } from 'react';
import { eventBus } from '../../utils/eventBus';
import { selectionManager } from '../../utils/selectionManager';

export default function InferenceTimeline() {
  const [currentLayer, setCurrentLayer] = useState(0);
  const [status, setStatus] = useState('paused');
  const [breakpointLayer, setBreakpointLayer] = useState(5);
  const [patchesCount, setPatchesCount] = useState(0);

  const totalLayers = 12;

  const handleStep = () => {
    if (currentLayer < totalLayers) {
      const next = currentLayer + 1;
      setCurrentLayer(next);
      selectionManager.setLayer(next);
      eventBus.emit('timeline:event', { type: 'step', label: `Stepped forward to Layer ${next}` });
      if (next === breakpointLayer) {
        setStatus('paused');
        eventBus.emit('timeline:event', { type: 'breakpoint', label: `Hit breakpoint at Layer ${next}` });
      }
    } else {
      setStatus('finished');
    }
  };

  const handleContinue = () => {
    setStatus('running');
    let layer = currentLayer;
    while (layer < totalLayers) {
      layer++;
      if (layer === breakpointLayer) {
        setCurrentLayer(layer);
        selectionManager.setLayer(layer);
        setStatus('paused');
        eventBus.emit('timeline:event', { type: 'breakpoint', label: `Hit breakpoint at Layer ${layer}` });
        return;
      }
    }
    setCurrentLayer(totalLayers);
    selectionManager.setLayer(totalLayers);
    setStatus('finished');
  };

  const handleApplyPatch = () => {
    setPatchesCount((prev) => prev + 1);
    eventBus.emit('timeline:event', { type: 'patch', label: `Applied replacement patch at Layer ${currentLayer}` });
  };

  return (
    <div className="panel-content inference-timeline-panel" data-testid="inference-timeline">
      <h4>Inference Stepper & Neural Timeline</h4>
      <p className="hint">Step through forward passes, inspect intermediate layers, set breakpoints, and patch activations.</p>

      <div className="stepper-controls" style={{ display: 'flex', gap: '8px', marginBottom: '14px' }}>
        <button className="btn btn-primary btn-sm" onClick={handleStep} disabled={status === 'finished'}>
          Step Layer ({currentLayer + 1}/{totalLayers})
        </button>
        <button className="btn btn-secondary btn-sm" onClick={handleContinue} disabled={status === 'finished'}>
          Continue to Breakpoint
        </button>
        <button className="btn btn-secondary btn-sm" onClick={handleApplyPatch}>
          + Apply Patch at L{currentLayer}
        </button>
        <span style={{ marginLeft: 'auto', fontSize: '12px', color: 'var(--text-dim)' }}>
          Status: <strong>{status.toUpperCase()}</strong> | Active Patches: <strong>{patchesCount}</strong>
        </span>
      </div>

      <div className="layer-timeline-stepper">
        <div
          className={`timeline-step-node ${currentLayer === 0 ? 'active' : ''}`}
          onClick={() => { setCurrentLayer(0); selectionManager.setLayer(0); }}
        >
          Emb
        </div>
        {Array.from({ length: totalLayers }).map((_, idx) => {
          const layerNum = idx + 1;
          const isCurrent = currentLayer === layerNum;
          const isBreakpoint = breakpointLayer === layerNum;
          return (
            <div
              key={layerNum}
              className={`timeline-step-node ${isCurrent ? 'active' : ''} ${isBreakpoint ? 'breakpoint' : ''}`}
              onClick={() => { setCurrentLayer(layerNum); selectionManager.setLayer(layerNum); }}
            >
              L{layerNum}
            </div>
          );
        })}
        <div
          className={`timeline-step-node ${currentLayer === totalLayers ? 'active' : ''}`}
          onClick={() => { setCurrentLayer(totalLayers); selectionManager.setLayer(totalLayers); }}
        >
          Pred
        </div>
      </div>
    </div>
  );
}
