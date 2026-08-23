import React, { useState } from 'react';
import './GpuSettings.css';

export const GpuSettings = ({ settings = {}, onChange }) => {
  const [gpuEnabled, setGpuEnabled] = useState(settings.gpuEnabled ?? false);
  const [device, setDevice] = useState(settings.device || 'cuda:0');
  const [precision, setPrecision] = useState(settings.precision || 'fp16');
  const [vramLimitPercent, setVramLimitPercent] = useState(settings.vramLimitPercent || 85);
  const [fallbackCpu, setFallbackCpu] = useState(settings.fallbackCpu ?? true);

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ gpuEnabled, device, precision, vramLimitPercent, fallbackCpu, ...patch });
    }
  };

  return (
    <div className="gpu-settings settings-page-container">
      <div className="settings-page-header">
        <h1>GPU & Hardware Acceleration</h1>
        <p>Configure CUDA/MPS device selection, inference precision, and VRAM allocation limits.</p>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="gpu-enabled-toggle"
            type="checkbox"
            checked={gpuEnabled}
            onChange={(e) => {
              setGpuEnabled(e.target.checked);
              handleUpdate({ gpuEnabled: e.target.checked });
            }}
          />
          <span>Enable GPU Acceleration (PyTorch CUDA / Apple Silicon MPS)</span>
        </label>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="gpu-device-select">Compute Device Target</label>
        <select
          id="gpu-device-select"
          data-testid="gpu-device-select"
          className="settings-select"
          disabled={!gpuEnabled}
          value={device}
          onChange={(e) => {
            setDevice(e.target.value);
            handleUpdate({ device: e.target.value });
          }}
        >
          <option value="cuda:0">NVIDIA CUDA GPU 0 (Default)</option>
          <option value="cuda:1">NVIDIA CUDA GPU 1</option>
          <option value="mps">Apple Silicon MPS</option>
          <option value="cpu">CPU Only (No Acceleration)</option>
        </select>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="gpu-precision-select">Execution Precision Codec</label>
        <select
          id="gpu-precision-select"
          data-testid="gpu-precision-select"
          className="settings-select"
          value={precision}
          onChange={(e) => {
            setPrecision(e.target.value);
            handleUpdate({ precision: e.target.value });
          }}
        >
          <option value="fp32">FP32 Full Precision (High VRAM)</option>
          <option value="fp16">FP16 Half Precision (Recommended)</option>
          <option value="bf16">BF16 Bfloat16 (Ampere / Ada / Hopper)</option>
          <option value="int8">INT8 Quantization (Low Memory Mode)</option>
        </select>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="vram-limit-slider">
          Max VRAM Allocation Budget: {vramLimitPercent}%
        </label>
        <input
          id="vram-limit-slider"
          data-testid="vram-limit-slider"
          type="range"
          min={30}
          max={95}
          step={5}
          className="settings-slider"
          value={vramLimitPercent}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10);
            setVramLimitPercent(val);
            handleUpdate({ vramLimitPercent: val });
          }}
        />
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="fallback-cpu-toggle"
            type="checkbox"
            checked={fallbackCpu}
            onChange={(e) => {
              setFallbackCpu(e.target.checked);
              handleUpdate({ fallbackCpu: e.target.checked });
            }}
          />
          <span>Automatically fall back to CPU if GPU runs out of VRAM (CUDA OOM)</span>
        </label>
      </div>
    </div>
  );
};
