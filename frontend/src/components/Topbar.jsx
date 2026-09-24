import React from 'react';
import { Brain, PanelLeft, Sidebar, Share2, Store, FileUp, Command, CircleDot } from 'lucide-react';
import { darkColors } from '../design/tokens/colors';
import { spacing } from '../design/tokens/spacing';
import { radii } from '../design/tokens/radii';
import { typography } from '../design/tokens/typography';
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
      ? '•••'
      : 'Offline';

  const statusColor = pythonStatus === 'connected'
    ? darkColors.success
    : pythonStatus === 'connecting'
      ? darkColors.warning
      : darkColors.textTertiary;

  return (
    <header
      className="topbar"
      style={{
        backgroundColor: darkColors.bgSecondary,
        borderBottom: `1px solid ${darkColors.borderPrimary}`,
        height: spacing.navbarHeight,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: `0 ${spacing[4]}`,
        gridArea: 'topbar',
      }}
    >
      {/* Left Side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: spacing[2] }}>
        {/* Toggle Buttons */}
        <button
          onClick={onToggleActivity}
          title={activityCollapsed ? 'Expand activity bar' : 'Collapse activity bar'}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: radii.sm,
            border: 'none',
            background: 'transparent',
            color: darkColors.textTertiary,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'background 0.15s ease, color 0.15s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = darkColors.bgHover;
            e.currentTarget.style.color = darkColors.textPrimary;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.color = darkColors.textTertiary;
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
            color: darkColors.textTertiary,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'background 0.15s ease, color 0.15s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = darkColors.bgHover;
            e.currentTarget.style.color = darkColors.textPrimary;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.color = darkColors.textTertiary;
          }}
        >
          <Sidebar size={16} strokeWidth={1.75} />
        </button>

        {/* Breadcrumb */}
        <span
          style={{
            fontFamily: typography.fontFamilySans,
            fontSize: typography.fontSizeSm,
            fontWeight: typography.fontWeightMedium,
            color: darkColors.textSecondary,
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {crumb}
        </span>
      </div>

      {/* Right Side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: spacing[2] }}>
        {/* Status Indicator */}
        <div
          data-testid="python-status"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: spacing[1.5],
            fontSize: typography.fontSizeXs,
            color: darkColors.textTertiary,
            padding: `${spacing[1.5]} ${spacing[3]}`,
            background: darkColors.bgPrimary,
            border: `1px solid ${darkColors.borderPrimary}`,
            borderRadius: radii.pill,
            whiteSpace: 'nowrap',
          }}
        >
          <CircleDot size={8} color={statusColor} />
          <span>{statusLabel}</span>
        </div>

        {/* Action Buttons */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onOpenDiscoveryMemory}
          leftIcon={<Brain size={14} />}
        >
          Mem
        </Button>

        <Button
          variant="ghost"
          size="sm"
          onClick={onToggleDock}
          leftIcon={<PanelLeft size={14} />}
        >
          Dock
        </Button>

        <Button
          variant="ghost"
          size="sm"
          onClick={onOpenShare}
          leftIcon={<Share2 size={14} />}
        >
          Share
        </Button>

        <Button
          variant="ghost"
          size="sm"
          onClick={onOpenMarketplace}
          leftIcon={<Store size={14} />}
        >
          Market
        </Button>

        <Button
          variant="ghost"
          size="sm"
          onClick={onOpenPublication}
          leftIcon={<FileUp size={14} />}
        >
          Publish
        </Button>

        {/* Command Palette */}
        <button
          onClick={onToggleCmdPalette}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: spacing[1.5],
            padding: `${spacing[1.5]} ${spacing[3]}`,
            background: darkColors.primaryBg,
            border: `1px solid ${darkColors.primaryBorder}`,
            borderRadius: radii.md,
            color: darkColors.primary,
            fontFamily: typography.fontFamilySans,
            fontSize: typography.fontSizeXs,
            fontWeight: typography.fontWeightMedium,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = darkColors.primary;
            e.currentTarget.style.color = darkColors.textOnPrimary;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = darkColors.primaryBg;
            e.currentTarget.style.color = darkColors.primary;
          }}
        >
          <Command size={14} />
          <span>Cmd+K</span>
        </button>
      </div>
    </header>
  );
}
