import React, { useState } from 'react';
import './DebuggerSettings.css';

export const DebuggerSettings = ({ settings = {}, onChange }) => {
  const [breakOnAnomaly, setBreakOnAnomaly] = useState(settings.breakOnAnomaly ?? true);
  const [maxTraceDepth, setMaxTraceDepth] = useState(settings.maxTraceDepth || 64);
  const [stepDelayMs, setStepDelayMs] = useState(settings.stepDelayMs || 100);
  const [captureHiddenStates, setCaptureHiddenStates] = useState(settings.captureHiddenStates ?? true);

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ breakOnAnomaly, maxTraceDepth, stepDelayMs, captureHiddenStates, ...patch });
    }
  };

  return (
    <div className="debugger-settings settings-page-container">
      <div className="settings-page-header">
        <h1>Circuit Debugger Settings</h1>
        <p>Configure stepping cadence, breakpoint triggers, and computational graph trace depth.</p>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="break-anomaly-toggle"
            type="checkbox"
            checked={breakOnAnomaly}
            onChange={(e) => {
              setBreakOnAnomaly(e.target.checked);
              handleUpdate({ breakOnAnomaly: e.target.checked });
            }}
          />
          <span>Pause execution when activation anomalies (NaN / Inf / Vanishing Gradients) are detected</span>
        </label>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="max-trace-depth">Maximum Causal Trace Depth (Layers)</label>
        <input
          id="max-trace-depth"
          data-testid="max-trace-depth-input"
          type="number"
          className="settings-input"
          min={4}
          max={256}
          value={maxTraceDepth}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10) || 32;
            setMaxTraceDepth(val);
            handleUpdate({ maxTraceDepth: val });
          }}
        />
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="step-delay-slider">
          Interactive Stepping Interval: {stepDelayMs} ms
        </label>
        <input
          id="step-delay-slider"
          data-testid="step-delay-slider"
          type="range"
          min={0}
          max={1000}
          step={50}
          className="settings-slider"
          value={stepDelayMs}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10);
            setStepDelayMs(val);
            handleUpdate({ stepDelayMs: val });
          }}
        />
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="capture-hidden-toggle"
            type="checkbox"
            checked={captureHiddenStates}
            onChange={(e) => {
              setCaptureHiddenStates(e.target.checked);
              handleUpdate({ captureHiddenStates: e.target.checked });
            }}
          />
          <span>Capture full residual stream activations at every breakpoint</span>
        </label>
      </div>
    </div>
  );
};
