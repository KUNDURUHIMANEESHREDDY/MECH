import React from 'react';

export const WorkspaceCanvas: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
  return (
    <main className="shell-workspace" style={workspaceStyle}>
      {children}
    </main>
  );
};

const workspaceStyle: React.CSSProperties = {
  flex: 1,
  minWidth: 0,
  background: 'var(--color-canvas, #ffffff)',
  position: 'relative',
  overflow: 'hidden',
};
