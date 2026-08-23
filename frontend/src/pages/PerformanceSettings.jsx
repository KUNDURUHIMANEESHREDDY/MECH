import React, { useState } from 'react';
import './PerformanceSettings.css';

export const PerformanceSettings = ({ settings = {}, onChange }) => {
  const [workerConcurrency, setWorkerConcurrency] = useState(settings.workerConcurrency || 4);
  const [batchSize, setBatchSize] = useState(settings.batchSize || 16);
  const [vectorSearchParallelism, setVectorSearchParallelism] = useState(settings.vectorSearchParallelism ?? true);
  const [memoryMappedIo, setMemoryMappedIo] = useState(settings.memoryMappedIo ?? true);
  const [jitCompilation, setJitCompilation] = useState(settings.jitCompilation ?? false);

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ workerConcurrency, batchSize, vectorSearchParallelism, memoryMappedIo, jitCompilation, ...patch });
    }
  };

  return (
    <div className="performance-settings settings-page-container">
      <div className="settings-page-header">
        <h1>Performance & Concurrency</h1>
        <p>Fine-tune thread worker pools, vector similarity indexing, and memory-mapped IO operations.</p>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="worker-concurrency-input">Parallel Worker Process Count</label>
        <input
          id="worker-concurrency-input"
          data-testid="worker-concurrency-input"
          type="number"
          className="settings-input"
          min={1}
          max={64}
          value={workerConcurrency}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10) || 4;
            setWorkerConcurrency(val);
            handleUpdate({ workerConcurrency: val });
          }}
        />
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="batch-size-input">Batch Evaluation Shard Size</label>
        <input
          id="batch-size-input"
          data-testid="batch-size-input"
          type="number"
          className="settings-input"
          min={1}
          max={512}
          value={batchSize}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10) || 16;
            setBatchSize(val);
            handleUpdate({ batchSize: val });
          }}
        />
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="vector-search-toggle"
            type="checkbox"
            checked={vectorSearchParallelism}
            onChange={(e) => {
              setVectorSearchParallelism(e.target.checked);
              handleUpdate({ vectorSearchParallelism: e.target.checked });
            }}
          />
          <span>Enable multi-threaded vector similarity search across hypothesis embeddings</span>
        </label>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="mmap-io-toggle"
            type="checkbox"
            checked={memoryMappedIo}
            onChange={(e) => {
              setMemoryMappedIo(e.target.checked);
              handleUpdate({ memoryMappedIo: e.target.checked });
            }}
          />
          <span>Enable zero-copy Memory-Mapped I/O (mmap) for weight tensors</span>
        </label>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="jit-compile-toggle"
            type="checkbox"
            checked={jitCompilation}
            onChange={(e) => {
              setJitCompilation(e.target.checked);
              handleUpdate({ jitCompilation: e.target.checked });
            }}
          />
          <span>Enable PyTorch TorchScript / `torch.compile` JIT acceleration</span>
        </label>
      </div>
    </div>
  );
};
