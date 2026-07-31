import React from 'react';
import { Cpu, Gauge, MemoryStick, Hash } from 'lucide-react';

interface Props {
  modelName: string;
  gpuUtil: number;
  memoryUtil: number;
  tokenCount: number;
  darkMode: boolean;
}

export function StatusBar({ modelName, gpuUtil, memoryUtil, tokenCount }: Props) {
  return (
    <div className="status-bar">
      <span className="status-item" style={{ fontWeight: 600 }}>
        <span className="status-icon"><Cpu size={12} /></span> Model: {modelName}
      </span>
      <span className="status-item">
        <span className="status-icon"><Gauge size={12} /></span> GPU: {(gpuUtil * 100).toFixed(0)}%
      </span>
      <span className="status-item">
        <span className="status-icon"><MemoryStick size={12} /></span> Memory: {(memoryUtil * 100).toFixed(0)}%
      </span>
      <span className="status-item">
        <span className="status-icon"><Hash size={12} /></span> Tokens: {tokenCount}
      </span>
    </div>
  );
}
