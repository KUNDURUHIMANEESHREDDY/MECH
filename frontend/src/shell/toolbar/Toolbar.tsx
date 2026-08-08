import React, { useState, useEffect } from 'react';
import { Brain, Search, Play, Download, Settings, User, Activity, PanelLeft, Sidebar } from 'lucide-react';
import { useUIStore } from '../../shared/stores/ui';
import { useWorkspaceStore } from '../../shared/stores/workspace';

interface ToolbarProps {
  onOpenCommandPalette: () => void;
  pythonStatus?: boolean;
}

export const Toolbar: React.FC<ToolbarProps> = ({ onOpenCommandPalette, pythonStatus = true }) => {
  const crumb = useUIStore((s) => s.crumb);
  const activityCollapsed = useUIStore((s) => s.activityCollapsed);
  const sidebarCollapsed = useUIStore((s) => s.sidebarCollapsed);
  const toggleActivity = useUIStore((s) => s.toggleActivity);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);

  return (
    <header className="shell-toolbar" style={toolbarStyle}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '14px' }}>
          <Brain size={18} style={{ color: 'var(--color-primary, #0066cc)' }} />
          <span>MECH</span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 400, background: 'var(--bg-elev-2)', padding: '2px 6px', borderRadius: '4px' }}>
            Canvas
          </span>
        </div>

        <button
          onClick={toggleActivity}
          title={activityCollapsed ? 'Expand activity bar' : 'Collapse activity bar'}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            border: 'none',
            background: 'transparent',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <PanelLeft size={16} strokeWidth={1.75} />
        </button>

        <button
          onClick={toggleSidebar}
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            border: 'none',
            background: 'transparent',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Sidebar size={16} strokeWidth={1.75} />
        </button>

        <span
          data-testid="topbar-crumb"
          style={{
            fontFamily: 'var(--font)',
            fontSize: '13px',
            fontWeight: 600,
            letterSpacing: '-0.2px',
            color: 'var(--text)',
          }}
        >
          {crumb}
        </span>
      </div>

      <button onClick={onOpenCommandPalette} style={searchBtnStyle}>
        <Search size={14} />
        <span>Search resources, panels, commands...</span>
        <kbd style={kbdStyle}>⌘K</kbd>
      </button>

      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <button style={actionBtnStyle} title="Run Activation Pass">
          <Play size={14} style={{ color: 'var(--color-primary)' }} />
          <span>Run Pass</span>
        </button>
        <button style={actionBtnStyle} title="Export Analysis State">
          <Download size={14} />
          <span>Export</span>
        </button>

        <div
          data-testid="python-status"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12px',
            color: pythonStatus ? 'var(--success, #1e8e3e)' : 'var(--danger, #d93025)',
            background: 'var(--bg-elev-2)',
            padding: '4px 8px',
            borderRadius: '6px',
          }}
        >
          <Activity size={12} />
          <span>{pythonStatus ? 'Python Ready' : 'Python Offline'}</span>
        </div>

        <button style={iconBtnStyle} title="Workspace Settings">
          <Settings size={15} />
        </button>
        <button style={iconBtnStyle} title="User Profile">
          <User size={15} />
        </button>
      </div>
    </header>
  );
};

const toolbarStyle: React.CSSProperties = {
  height: 'var(--topbar-h, 44px)',
  background: 'var(--color-canvas, #ffffff)',
  borderBottom: '1px solid var(--border, #e0e0e0)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '0 16px',
  userSelect: 'none',
  zIndex: 10,
};

const searchBtnStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  width: '320px',
  background: 'var(--bg-elev-2, #f5f5f5)',
  border: '1px solid var(--border, #e0e0e0)',
  borderRadius: '8px',
  padding: '5px 12px',
  fontSize: '12px',
  color: 'var(--text-muted, #7a7a7a)',
  cursor: 'pointer',
};

const kbdStyle: React.CSSProperties = {
  marginLeft: 'auto',
  background: 'var(--color-canvas, #ffffff)',
  border: '1px solid var(--border, #e0e0e0)',
  borderRadius: '4px',
  padding: '1px 5px',
  fontSize: '10px',
  fontWeight: 600,
};

const actionBtnStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
  background: 'var(--bg-elev-2, #f5f5f5)',
  border: '1px solid var(--border, #e0e0e0)',
  borderRadius: '6px',
  padding: '5px 10px',
  fontSize: '12px',
  fontWeight: 500,
  color: 'var(--text, #1d1d1f)',
  cursor: 'pointer',
};

const iconBtnStyle: React.CSSProperties = {
  background: 'transparent',
  border: 'none',
  padding: '6px',
  borderRadius: '6px',
  color: 'var(--text-dim, #333333)',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
};
