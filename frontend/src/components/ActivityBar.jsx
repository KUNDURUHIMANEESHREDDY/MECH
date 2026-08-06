import React from 'react';
import { LayoutGrid, Search, Bug, FileCode2, Settings, Sidebar } from 'lucide-react';
import { darkColors } from '../design/tokens/colors';
import { spacing } from '../design/tokens/spacing';
import { radii } from '../design/tokens/radii';
import { typography } from '../design/tokens/typography';

const TOP_ICONS = [
  { id: 'explorer', icon: LayoutGrid, label: 'Explorer', section: 'explorer' },
  { id: 'search', icon: Search, label: 'Search' },
  { id: 'debug', icon: Bug, label: 'Debug', section: 'debugger' },
  { id: 'code', icon: FileCode2, label: 'Code' },
];

const BOTTOM_ICONS = [
  { id: 'settings', icon: Settings, label: 'Settings', section: 'settings' },
];

export default function ActivityBar({ active, onSelect, collapsed, onToggle, onToggleSidebar }) {
  return (
    <div
      className={`activity-bar${collapsed ? ' collapsed' : ''}`}
      style={{
        backgroundColor: darkColors.bgSecondary,
        width: collapsed ? '0px' : spacing.activityBarWidth,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        paddingTop: spacing[2],
        paddingBottom: spacing[2],
        transition: 'width 0.2s ease',
        overflow: 'hidden',
        borderRight: `1px solid ${darkColors.borderPrimary}`,
      }}
    >
      {/* Top Section */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: spacing[2], flex: 1 }}>
        {/* Logo */}
        <div
          style={{
            width: '32px',
            height: '32px',
            borderRadius: radii.circle,
            background: darkColors.gradientPrimary,
            color: darkColors.textOnPrimary,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: typography.fontFamilySans,
            fontWeight: typography.fontWeightBold,
            fontSize: typography.fontSizeBase,
            marginBottom: spacing[3],
            cursor: 'pointer',
            userSelect: 'none',
            boxShadow: '0 2px 4px rgba(79, 70, 229, 0.2)',
          }}
        >
          M
        </div>

        {/* Top Icons */}
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
                borderRadius: radii.circle,
                border: 'none',
                background: isActive ? darkColors.primary : 'transparent',
                color: isActive ? darkColors.textOnPrimary : darkColors.textTertiary,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                boxShadow: isActive ? '0 2px 4px rgba(79, 70, 229, 0.2)' : 'none',
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.currentTarget.style.background = darkColors.bgHover;
                  e.currentTarget.style.color = darkColors.textPrimary;
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = darkColors.textTertiary;
                }
              }}
            >
              <Icon size={18} strokeWidth={1.75} />
            </button>
          );
        })}
      </div>

      {/* Bottom Section */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: spacing[2] }}>
        {/* Bottom Icons */}
        {BOTTOM_ICONS.map((item) => {
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
                borderRadius: radii.circle,
                border: 'none',
                background: isActive ? darkColors.primary : 'transparent',
                color: isActive ? darkColors.textOnPrimary : darkColors.textTertiary,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.currentTarget.style.background = darkColors.bgHover;
                  e.currentTarget.style.color = darkColors.textPrimary;
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  e.currentTarget.style.background = 'transparent';
                  e.currentTarget.style.color = darkColors.textTertiary;
                }
              }}
            >
              <Icon size={18} strokeWidth={1.75} />
            </button>
          );
        })}

        {/* Toggle Sidebar Button */}
        <button
          onClick={onToggleSidebar}
          title="Toggle sidebar"
          style={{
            width: '36px',
            height: '36px',
            borderRadius: radii.circle,
            border: 'none',
            background: 'transparent',
            color: darkColors.textTertiary,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.15s ease',
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
          <Sidebar size={18} strokeWidth={1.75} />
        </button>

        {/* Collapse Button */}
        <button
          onClick={onToggle}
          title={collapsed ? 'Expand activity bar' : 'Collapse activity bar'}
          style={{
            width: '36px',
            height: '36px',
            borderRadius: radii.circle,
            border: 'none',
            background: 'transparent',
            color: darkColors.textTertiary,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.15s ease',
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
          {collapsed ? <ChevronRight size={18} strokeWidth={1.75} /> : <ChevronLeft size={18} strokeWidth={1.75} />}
        </button>
      </div>
    </div>
  );
}

// Import Chevron icons
import { ChevronLeft, ChevronRight } from 'lucide-react';
