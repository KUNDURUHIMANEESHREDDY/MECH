import React, { useState } from 'react';
import { Terminal, ChevronUp, ChevronDown } from 'lucide-react';
import { colors, radii, spacing, typography } from '../design/tokens';

export default function ConsolePanel({ logs = [] }) {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <div
      className="console-panel"
      style={{
        backgroundColor: colors.surfacePearl,
        color: colors.ink,
        borderTop: `1px solid ${colors.hairline}`,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <div
        className="console-bar"
        onClick={() => setCollapsed(!collapsed)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: `${spacing.xs} ${spacing.md}`,
          borderBottom: `1px solid ${colors.border}`,
          fontFamily: typography.buttonUtility.fontFamily,
          fontSize: typography.buttonUtility.fontSize,
          color: colors.bodyMuted,
          cursor: 'pointer',
        }}
      >
        <span><Terminal size={13} /> Console</span>
        <span style={{ fontFamily: typography.finePrint.fontFamily, fontSize: typography.finePrint.fontSize }}>
          {collapsed ? <ChevronUp size={13} /> : <ChevronDown size={13} />} {logs.length} lines
        </span>
      </div>
      {!collapsed && (
        <div
          className="console-body"
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: spacing.sm,
            fontFamily: typography.body.fontFamily,
            fontSize: typography.body.fontSize,
            lineHeight: typography.body.lineHeight,
          }}
        >
          {logs.length === 0 && <div className="console-line" style={{ color: colors.bodyMuted }}>[System ready]</div>}
          {logs.map((line, i) => (
            <div
              key={i}
              className={'console-line' + (line.type ? ' ' + line.type : '')}
              style={{
                padding: `${spacing.xs} 0`,
                color:
                  line.type === 'error'
                    ? colors.danger
                    : line.type === 'warn'
                    ? colors.warning
                    : line.type === 'success'
                    ? colors.success
                    : colors.ink,
              }}
            >
              &gt; {line.text || line}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
