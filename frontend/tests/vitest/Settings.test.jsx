import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ThemeSettings } from '../../src/pages/ThemeSettings';
import { GpuSettings } from '../../src/pages/GpuSettings';
import { CacheSettings } from '../../src/pages/CacheSettings';
import { DebuggerSettings } from '../../src/pages/DebuggerSettings';
import { ModelsSettings } from '../../src/pages/ModelsSettings';
import { PathsSettings } from '../../src/pages/PathsSettings';
import { LoggingSettings } from '../../src/pages/LoggingSettings';
import { PerformanceSettings } from '../../src/pages/PerformanceSettings';
import { PluginsSettings } from '../../src/pages/PluginsSettings';
import { PythonSettings } from '../../src/pages/PythonSettings';

describe('Settings subpages', () => {
  it('ThemeSettings renders and fires onChange on theme selection', () => {
    const onChange = vi.fn();
    render(<ThemeSettings settings={{ theme: 'light' }} onChange={onChange} />);
    expect(screen.getByText('Theme & Appearance')).toBeInTheDocument();
    const select = screen.getByTestId('theme-select');
    expect(select.value).toBe('light');
    fireEvent.change(select, { target: { value: 'dark' } });
    expect(onChange).toHaveBeenCalled();
  });

  it('GpuSettings renders hardware toggles and precision selector', () => {
    const onChange = vi.fn();
    render(<GpuSettings settings={{ gpuEnabled: false }} onChange={onChange} />);
    expect(screen.getByText('GPU & Hardware Acceleration')).toBeInTheDocument();
    const toggle = screen.getByTestId('gpu-enabled-toggle');
    fireEvent.click(toggle);
    expect(onChange).toHaveBeenCalled();
  });

  it('CacheSettings renders cache path input and purge actions', () => {
    const onChange = vi.fn();
    render(<CacheSettings settings={{ maxCacheSizeMb: 4096 }} onChange={onChange} />);
    expect(screen.getByText('Cache Settings')).toBeInTheDocument();
    expect(screen.getByTestId('max-cache-size').value).toBe('4096');
    const purgeBtn = screen.getByTestId('clear-cache-btn');
    fireEvent.click(purgeBtn);
    expect(screen.getByText(/Cache purged successfully/i)).toBeInTheDocument();
  });

  it('DebuggerSettings renders anomaly breakpoint and stepping slider', () => {
    const onChange = vi.fn();
    render(<DebuggerSettings onChange={onChange} />);
    expect(screen.getByText('Circuit Debugger Settings')).toBeInTheDocument();
    const slider = screen.getByTestId('step-delay-slider');
    fireEvent.change(slider, { target: { value: '250' } });
    expect(onChange).toHaveBeenCalled();
  });

  it('ModelsSettings renders model architecture catalog and HF token input', () => {
    const onChange = vi.fn();
    render(<ModelsSettings onChange={onChange} />);
    expect(screen.getByText('Model Architectures & Weights')).toBeInTheDocument();
    const select = screen.getByTestId('model-select');
    fireEvent.change(select, { target: { value: 'meta-llama/Llama-3.2-1B' } });
    expect(onChange).toHaveBeenCalled();
  });

  it('PathsSettings renders workspace directory inputs and reset button', () => {
    const onChange = vi.fn();
    render(<PathsSettings onChange={onChange} />);
    expect(screen.getByText('Workspace & File Paths')).toBeInTheDocument();
    const resetBtn = screen.getByTestId('reset-paths-btn');
    fireEvent.click(resetBtn);
    expect(screen.getByText(/Paths reset to default/i)).toBeInTheDocument();
  });

  it('LoggingSettings renders log level selector and flush buffer trigger', () => {
    const onChange = vi.fn();
    render(<LoggingSettings onChange={onChange} />);
    expect(screen.getByText('Logging & Diagnostics')).toBeInTheDocument();
    const flushBtn = screen.getByTestId('flush-logs-btn');
    fireEvent.click(flushBtn);
    expect(screen.getByText(/Active log buffers flushed/i)).toBeInTheDocument();
  });

  it('PerformanceSettings renders worker pool count and mmap toggle', () => {
    const onChange = vi.fn();
    render(<PerformanceSettings onChange={onChange} />);
    expect(screen.getByText('Performance & Concurrency')).toBeInTheDocument();
    const input = screen.getByTestId('worker-concurrency-input');
    fireEvent.change(input, { target: { value: '8' } });
    expect(onChange).toHaveBeenCalled();
  });

  it('PluginsSettings renders extension scan button and plugin catalog', () => {
    const onChange = vi.fn();
    render(<PluginsSettings onChange={onChange} />);
    expect(screen.getByText('Plugins & Extensions')).toBeInTheDocument();
    const scanBtn = screen.getByTestId('scan-plugins-btn');
    fireEvent.click(scanBtn);
    expect(screen.getByText(/Found 3 installed plugin packages/i)).toBeInTheDocument();
  });

  it('PythonSettings renders interpreter path and connection test button', () => {
    const onChange = vi.fn();
    render(<PythonSettings onChange={onChange} />);
    expect(screen.getByText('Python Backend & Sidecar')).toBeInTheDocument();
    const testBtn = screen.getByTestId('test-python-btn');
    fireEvent.click(testBtn);
    expect(screen.getByText(/Checking Python sidecar health/i)).toBeInTheDocument();
  });
});
