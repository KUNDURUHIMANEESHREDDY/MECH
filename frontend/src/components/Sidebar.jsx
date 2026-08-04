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
import { colors, radii, spacing, typography } from '../design/tokens';

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
        backgroundColor: colors.bgSidebar,
        width: collapsed ? '0px' : '264px',
        borderRight: `1px solid ${colors.border}`,
        display: 'flex',
        flexDirection: 'column',
        transition: 'width 0.2s ease',
        overflow: 'hidden',
        gridArea: 'sidebar',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: `${spacing.md} ${spacing.md} ${spacing.sm}`,
          borderBottom: `1px solid ${colors.border}`,
        }}
      >
        <h1
          style={{
            fontFamily: typography.displayLg.fontFamily,
            fontSize: typography.displayLg.fontSize,
            fontWeight: typography.displayLg.fontWeight,
            letterSpacing: typography.displayLg.letterSpacing,
            color: colors.ink,
            margin: 0,
          }}
        >
          Explorer
        </h1>
        <button
          onClick={onToggle}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          style={{
            width: '32px',
            height: '32px',
            borderRadius: radii.full,
            border: 'none',
            background: 'transparent',
            color: colors.inkMuted80,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
      {!collapsed && (
        <div style={{ padding: `${spacing.sm} ${spacing.md}`, flex: 1, overflowY: 'auto' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: spacing.xs,
              background: colors.canvas,
              borderRadius: radii.pill,
              padding: `${spacing.xs} ${spacing.md}`,
              marginBottom: spacing.md,
              border: `1px solid ${colors.border}`,
            }}
          >
            <Search size={13} color={colors.inkMuted48} />
            <input
              type="text"
              placeholder="Search views..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{
                border: 'none',
                outline: 'none',
                background: 'transparent',
                fontFamily: typography.body.fontFamily,
                fontSize: typography.body.fontSize,
                color: colors.ink,
                width: '100%',
                padding: '4px 0',
              }}
            />
          </div>
          {SECTIONS.map((section) => (
            <div key={section.title} style={{ marginBottom: spacing.md }}>
              <div
                style={{
                  fontFamily: typography.caption.fontFamily,
                  fontSize: typography.caption.fontSize,
                  fontWeight: typography.caption.fontWeight,
                  letterSpacing: typography.caption.letterSpacing,
                  color: colors.inkMuted48,
                  padding: `${spacing.xs} 0`,
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
                        gap: spacing.sm,
                        padding: `${spacing.xs} ${spacing.sm}`,
                        borderRadius: radii.sm,
                        cursor: 'pointer',
                        backgroundColor: isActive ? colors.accentSoft : 'transparent',
                        color: isActive ? colors.primary : colors.ink,
                        fontFamily: typography.body.fontFamily,
                        fontSize: typography.body.fontSize,
                        fontWeight: isActive ? typography.bodyStrong.fontWeight : typography.body.fontWeight,
                        transition: 'background-color 0.15s ease',
                      }}
                    >
                      {Icon && (
                        <span style={{ display: 'flex', alignItems: 'center' }}>
                          <Icon size={15} strokeWidth={1.75} />
                        </span>
                      )}
                      <span>{getLabel(key)}</span>
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
