import React from 'react';
import { Brain, PanelLeft, Sidebar, Share2, Store, FileUp, Command, CircleDot } from 'lucide-react';
import { colors, radii, spacing, typography } from '../design/tokens';
import { Button } from '../design/components/Button';

export default function Topbar({
  crumb,
  pythonStatus,
  onToggleDock,
  onToggleCmdPalette,
  onOpenShare,
  onOpenMarketplace,
  onOpenPublication,
  onOpenDiscoveryMemory,
  sidebarCollapsed,
  onToggleSidebar,
  activityCollapsed,
  onToggleActivity,
}) {
  const statusLabel = pythonStatus === 'connected'
    ? 'Connected'
    : pythonStatus === 'connecting'
      ? '…'
      : 'Offline';

  return (
    <header
      className="topbar"
      style={{
        backgroundColor: colors.canvas,
        borderBottom: `1px solid ${colors.hairline}`,
        height: '46px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: `0 ${spacing.lg}`,
        gridArea: 'topbar',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: spacing.sm }}>
        <button
          onClick={onToggleActivity}
          title={activityCollapsed ? 'Expand activity bar' : 'Collapse activity bar'}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: radii.sm,
            border: 'none',
            background: 'transparent',
            color: colors.inkMuted80,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <PanelLeft size={16} strokeWidth={1.75} />
        </button>
        <button
          onClick={onToggleSidebar}
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: radii.sm,
            border: 'none',
            background: 'transparent',
            color: colors.inkMuted80,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Sidebar size={16} strokeWidth={1.75} />
        </button>
        <span
          style={{
            fontFamily: typography.tagline.fontFamily,
            fontSize: typography.tagline.fontSize,
            fontWeight: typography.tagline.fontWeight,
            letterSpacing: typography.tagline.letterSpacing,
            color: colors.ink,
          }}
        >
          {crumb}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: spacing.xs }}>
        <Button variant="dark-utility" onClick={onOpenDiscoveryMemory} style={{ padding: `${spacing.xs} ${spacing.sm}` }}>
          <Brain size={14} /> Mem
        </Button>
        <Button variant="dark-utility" onClick={onToggleDock} style={{ padding: `${spacing.xs} ${spacing.sm}` }}>
          <PanelLeft size={14} /> Dock
        </Button>
        <Button variant="dark-utility" onClick={onOpenShare} style={{ padding: `${spacing.xs} ${spacing.sm}` }}>
          <Share2 size={14} /> Share
        </Button>
        <Button variant="dark-utility" onClick={onOpenMarketplace} style={{ padding: `${spacing.xs} ${spacing.sm}` }}>
          <Store size={14} /> Market
        </Button>
        <Button variant="dark-utility" onClick={onOpenPublication} style={{ padding: `${spacing.xs} ${spacing.sm}` }}>
          <FileUp size={14} /> Export
        </Button>
        <Button variant="dark-utility" onClick={onToggleCmdPalette} style={{ padding: `${spacing.xs} ${spacing.sm}` }}>
          <Command size={13} /> P
        </Button>
        <div
          data-testid="python-status"
          data-status={pythonStatus}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: spacing.xs,
            paddingLeft: spacing.sm,
            marginLeft: spacing.xs,
            borderLeft: `1px solid ${colors.border}`,
          }}
        >
          <span
            className={`dot ${pythonStatus === 'connected' ? 'connected' : pythonStatus === 'connecting' ? 'connecting' : ''}`}
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: pythonStatus === 'connected' ? colors.success : pythonStatus === 'connecting' ? colors.warning : colors.inkMuted48,
            }}
          />
          <span style={{ fontFamily: typography.navLink.fontFamily, fontSize: typography.navLink.fontSize, color: colors.inkMuted80 }}>
            {statusLabel}
          </span>
        </div>
      </div>
    </header>
  );
}
