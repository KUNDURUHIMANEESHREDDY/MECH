import React from 'react';
import { Cpu, Gauge, MemoryStick, Hash } from 'lucide-react';
import { colors, radii, spacing, typography } from '../design/tokens';

interface Props {
  modelName: string;
  gpuUtil: number;
  memoryUtil: number;
  tokenCount: number;
}

export function StatusBar({ modelName, gpuUtil, memoryUtil, tokenCount }: Props) {
  return (
    <footer
      className="statusbar"
      style={{
        backgroundColor: colors.bgSidebar,
        borderTop: `1px solid ${colors.border}`,
        height: '26px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: `0 ${spacing.md}`,
        fontFamily: typography.finePrint.fontFamily,
        fontSize: typography.finePrint.fontSize,
        color: colors.inkMuted48,
      }}
    >
      <span style={{ fontWeight: typography.finePrint.fontWeight }}>{modelName}</span>
      <span style={{ display: 'flex', gap: spacing.lg }}>
        <span>GPU: {Math.round(gpuUtil * 100)}%</span>
        <span>Memory: {Math.round(memoryUtil * 100)}%</span>
        <span>Tokens: {tokenCount}</span>
      </span>
    </footer>
  );
}
