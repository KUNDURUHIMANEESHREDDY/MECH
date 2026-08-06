import React, { useState } from 'react';
import {
  Network,
  Workflow,
  PanelsTopLeft,
  BrainCircuit,
  Share2,
  Waypoints,
  Boxes,
  Terminal,
  Bug,
  Hammer,
  Gauge,
  ListChecks,
  FlaskConical,
  History,
  FileText,
  GitBranch,
  Scale,
  TrendingUp,
  HeartPulse,
  Rocket,
  Puzzle,
  BookOpen,
  ClipboardList,
  CopyCheck,
  Settings,
  ScrollText,
  FolderKanban,
  Clock,
  Search,
  ChevronLeft,
  ChevronRight,
  Activity,
} from 'lucide-react';
import { darkColors } from '../design/tokens/colors';
import { spacing } from '../design/tokens/spacing';
import { radii } from '../design/tokens/radii';
import { typography } from '../design/tokens/typography';

const SECTIONS = [
  {
    title: 'Explore',
    items: ['explorer', 'gpt2', 'transformer', 'workspace', 'neuralexplorer', 'knowledgegraph', 'circuitexplorer'],
  },
  {
    title: 'Develop',
    items: ['models', 'prompts', 'debugger', 'build', 'benchmark', 'benchmarksuite'],
  },
  {
    title: 'Research',
    items: ['experiments', 'sessions', 'reports', 'reasoning', 'evidencefusion', 'analytics', 'health'],
  },
  {
    title: 'General',
    items: ['campaigns', 'plugins', 'notebook', 'labnotebook', 'reproduction', 'settings', 'logging', 'projects', 'recent'],
  },
];

const PAGE_LABELS = {
  explorer: 'Model Explorer',
  gpt2: 'GPT-2 Live',
  transformer: 'Transformer Visualizer',
  workspace: 'Workspace',
  neuralexplorer: 'Neural Explorer',
  knowledgegraph: 'Knowledge Graph',
  circuitexplorer: 'Circuit Explorer',
  models: 'Models',
  prompts: 'Prompts',
  debugger: 'Debugger',
  build: 'Build Log',
  benchmark: 'Benchmark',
  benchmarksuite: 'Benchmark Suite',
  experiments: 'Experiments',
  sessions: 'Sessions',
  reports: 'Reports',
  reasoning: 'Reasoning Trace',
  evidencefusion: 'Evidence Fusion',
  analytics: 'Analytics',
  health: 'Health',
  campaigns: 'Campaigns',
  plugins: 'Plugins',
  notebook: 'Research Notebook',
  labnotebook: 'Lab Notebook',
  reproduction: 'Paper Reproduction',
  settings: 'Settings',
  logging: 'Logging',
  projects: 'Projects',
  recent: 'Recent Files',
};

const ICONS = {
  explorer: Network,
  gpt2: Activity,
  transformer: Workflow,
  workspace: PanelsTopLeft,
  neuralexplorer: BrainCircuit,
  knowledgegraph: Share2,
  circuitexplorer: Waypoints,
  models: Boxes,
  prompts: Terminal,
  debugger: Bug,
  build: Hammer,
  benchmark: Gauge,
  benchmarksuite: ListChecks,
  experiments: FlaskConical,
  sessions: History,
  reports: FileText,
  reasoning: GitBranch,
  evidencefusion: Scale,
  analytics: TrendingUp,
  health: HeartPulse,
  campaigns: Rocket,
  plugins: Puzzle,
  notebook: BookOpen,
  labnotebook: ClipboardList,
  reproduction: CopyCheck,
  settings: Settings,
  logging: ScrollText,
  projects: FolderKanban,
  recent: Clock,
};

export default function Sidebar({ pages, active, onSelect, collapsed, onToggle }) {
  const [query, setQuery] = useState('');
  const getLabel = (key) => pages[key]?.label || PAGE_LABELS[key] || key;
  const matches = (key) => getLabel(key).toLowerCase().includes(query.trim().toLowerCase());

  return (
    <div
      className={`sidebar${collapsed ? ' collapsed' : ''}`}
      style={{
        backgroundColor: darkColors.bgSecondary,
        width: collapsed ? spacing.sidebarCollapsedWidth : spacing.sidebarWidth,
        borderRight: `1px solid ${darkColors.borderPrimary}`,
        display: 'flex',
        flexDirection: 'column',
        transition: 'width 0.2s ease',
        overflow: 'hidden',
      }}
    >
      {/* Sidebar Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: `${spacing[3]} ${spacing[3.5]}`,
          borderBottom: `1px solid ${darkColors.borderPrimary}`,
          flexShrink: 0,
        }}
      >
        {!collapsed && (
          <h1
            style={{
              fontFamily: typography.fontFamilySans,
              fontSize: typography.fontSizeSm,
              fontWeight: typography.fontWeightSemibold,
              letterSpacing: typography.letterSpacingWider,
              color: darkColors.textTertiary,
              textTransform: 'uppercase',
              margin: 0,
            }}
          >
            Explorer
          </h1>
        )}
        <button
          onClick={onToggle}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: radii.circle,
            border: 'none',
            background: collapsed ? darkColors.bgHover : 'transparent',
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
            e.currentTarget.style.background = collapsed ? darkColors.bgHover : 'transparent';
            e.currentTarget.style.color = darkColors.textTertiary;
          }}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* Sidebar Body */}
      {!collapsed && (
        <div style={{ padding: `${spacing[2]} 0 ${spacing[3]}`, flex: 1, overflowY: 'auto' }}>
          {/* Search */}
          <div style={{ padding: `0 ${spacing[3]} ${spacing[3]}` }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: spacing[2],
                background: darkColors.bgPrimary,
                borderRadius: radii.md,
                padding: `${spacing[2]} ${spacing[3]}`,
                border: `1px solid ${darkColors.borderPrimary}`,
              }}
            >
              <Search size={14} color={darkColors.textTertiary} />
              <input
                type="text"
                placeholder="Search views..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{
                  border: 'none',
                  outline: 'none',
                  background: 'transparent',
                  fontFamily: typography.fontFamilySans,
                  fontSize: typography.fontSizeSm,
                  color: darkColors.textPrimary,
                  width: '100%',
                }}
              />
            </div>
          </div>

          {/* Navigation Sections */}
          {SECTIONS.map((section) => (
            <div key={section.title} style={{ marginBottom: spacing[3] }}>
              <div
                style={{
                  fontFamily: typography.fontFamilySans,
                  fontSize: typography.fontSizeXs,
                  fontWeight: typography.fontWeightSemibold,
                  letterSpacing: typography.letterSpacingWider,
                  color: darkColors.textTertiary,
                  padding: `${spacing[2]} ${spacing[3]}`,
                  textTransform: 'uppercase',
                }}
              >
                {section.title}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {section.items.filter(matches).map((key) => {
                  const Icon = ICONS[key];
                  const isActive = active === key;
                  return (
                    <div
                      key={key}
                      role="button"
                      tabIndex={0}
                      className={'nav-item' + (isActive ? ' active' : '')}
                      data-testid={`nav-${key}`}
                      onClick={() => onSelect(key)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') onSelect(key);
                      }}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: spacing[2],
                        padding: `${spacing[2]} ${spacing[3]}`,
                        borderRadius: radii.sm,
                        cursor: 'pointer',
                        backgroundColor: isActive ? darkColors.primaryBg : 'transparent',
                        color: isActive ? darkColors.primary : darkColors.textSecondary,
                        fontFamily: typography.fontFamilySans,
                        fontSize: typography.fontSizeSm,
                        fontWeight: isActive ? typography.fontWeightSemibold : typography.fontWeightNormal,
                        transition: 'background-color 0.15s ease, color 0.15s ease',
                        margin: `0 ${spacing[1.5]}`,
                      }}
                      onMouseEnter={(e) => {
                        if (!isActive) {
                          e.currentTarget.style.backgroundColor = darkColors.bgHover;
                          e.currentTarget.style.color = darkColors.textPrimary;
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!isActive) {
                          e.currentTarget.style.backgroundColor = 'transparent';
                          e.currentTarget.style.color = darkColors.textSecondary;
                        }
                      }}
                    >
                      {Icon && (
                        <span style={{ display: 'flex', alignItems: 'center' }}>
                          <Icon size={15} strokeWidth={1.75} />
                        </span>
                      )}
                      <span>{getLabel(key)}</span>
                      {isActive && (
                        <div
                          style={{
                            marginLeft: 'auto',
                            width: '2px',
                            height: '18px',
                            borderRadius: '2px',
                            background: darkColors.primary,
                          }}
                        />
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
