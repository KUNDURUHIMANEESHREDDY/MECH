import React from 'react';

interface Props {
  modelName: string;
  gpuUtil: number;
  memoryUtil: number;
  tokenCount: number;
  darkMode: boolean;
}

export function StatusBar({ modelName, gpuUtil, memoryUtil, tokenCount, darkMode }: Props) {
  const bg = darkMode ? '#111118' : '#e0e0e0';
  const fg = darkMode ? '#ccc' : '#333';

  return (
    <div style={{
      display: 'flex', gap: 20, padding: '6px 14px', background: bg, color: fg,
      fontSize: 11, fontFamily: 'monospace', alignItems: 'center', borderTop: darkMode ? '1px solid #2a2a3a' : '1px solid #ccc',
    }}>
      <span style={{ fontWeight: 600 }}>Model: {modelName}</span>
      <span>GPU: {(gpuUtil * 100).toFixed(0)}%</span>
      <span>Memory: {(memoryUtil * 100).toFixed(0)}%</span>
      <span>Tokens: {tokenCount}</span>
    </div>
  );
}
