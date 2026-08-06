import React from 'react';
import { LayoutGrid, Search, Bug, FileCode2, Settings, Sidebar } from 'lucide-react';
import { colors, radii, spacing, typography } from '../design/tokens';

const TOP_ICONS = [
  { id: 'explorer', icon: LayoutGrid, label: 'Explorer', section: 'explorer' },
  { id: 'search', icon: Search, label: 'Search' },
  { id: 'debug', icon: Bug, label: 'Debug', section: 'debugger' },
  { id: 'code', icon: FileCode2, label: 'Code' },
];

export default function ActivityBar({ active, onSelect, collapsed, onToggle, onToggleSidebar }) {
  return (
    <div
      className={`activity-bar${collapsed ? ' collapsed' : ''}`}
      style={{
        backgroundColor: colors.bgSidebar,
        width: collapsed ? '0px' : '52px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        paddingTop: spacing.sm,
        paddingBottom: spacing.sm,
        transition: 'width 0.2s ease',
        overflow: 'hidden',
        borderRight: `1px solid ${colors.hairline}`,
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: spacing.sm, flex: 1 }}>
        <div
          style={{
            width: '32px',
            height: '32px',
            borderRadius: radii.full,
            backgroundColor: colors.primary,
            color: colors.onPrimary,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: typography.bodyStrong.fontFamily,
            fontWeight: typography.bodyStrong.fontWeight,
            fontSize: typography.bodyStrong.fontSize,
            marginBottom: spacing.md,
          }}
        >
          M
        </div>
        {TOP_ICONS.map((item) => {
          const Icon = item.icon;
          const isActive = active === item.section;
          return (
            <button
              key={item.id}
              onClick={() => item.section && onSelect(item.section)}
              title={item.label}
              style={{
                width: '36px',
                height: '36px',
                borderRadius: radii.full,
                border: 'none',
                background: isActive ? colors.primary : 'transparent',
                color: isActive ? colors.onPrimary : colors.inkMuted48,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'background 0.15s ease',
              }}
            >
              <Icon size={18} strokeWidth={1.75} />
            </button>
          );
        })}
      </div>
      <button
        onClick={onToggleSidebar}
        title="Toggle sidebar"
        style={{
          width: '36px',
          height: '36px',
          borderRadius: radii.full,
          border: 'none',
          background: 'transparent',
          color: colors.inkMuted48,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Sidebar size={18} strokeWidth={1.75} />
      </button>
      <button
        onClick={onToggle}
        title={collapsed ? 'Expand activity bar' : 'Collapse activity bar'}
        style={{
          width: '36px',
          height: '36px',
          borderRadius: radii.full,
          border: 'none',
          background: 'transparent',
          color: colors.inkMuted48,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Settings size={18} strokeWidth={1.75} />
      </button>
    </div>
  );
}
