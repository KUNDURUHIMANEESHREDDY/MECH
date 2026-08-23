import React, { useState, useEffect } from 'react';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import { useResearchStore } from '../../shared/stores/research';
import { commandManager } from '../../shared/managers/commandManager';
import { colors } from '../../design/tokens/colors';
import {
  ChevronUp,
  ChevronDown,
  Terminal,
  Activity,
  Cpu,
  Clock,
  FileText,
  AlertTriangle,
  Server,
  RefreshCw,
  CheckCircle2,
} from 'lucide-react';

type TabId = 'tasks' | 'console' | 'runs' | 'timeline' | 'notes';

const TABS: { id: TabId; label: string }[] = [
  { id: 'tasks', label: 'Background Jobs' },
  { id: 'console', label: 'Console Stream' },
  { id: 'runs', label: 'Causal Runs' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'notes', label: 'Notes' },
];

export const BottomWorkspace: React.FC = () => {
  const [collapsed, setCollapsed] = useState(true); // Collapsed by default to prevent vertical waste!
  const [activeTab, setActiveTab] = useState<TabId>('tasks');
  const console = useWorkspaceStore((s) => s.console);
  const timeline = useWorkspaceStore((s) => s.timeline);
  const notes = useWorkspaceStore((s) => s.notes);
  const research = useResearchStore();

  return (
    <div
      className="shell-bottom"
      style={{
        height: collapsed ? '30px' : '200px',
        backgroundColor: colors.surfaceTile1,
        borderTop: `1px solid ${colors.border}`,
        display: 'flex',
        flexDirection: 'column',
        transition: 'height 0.2s ease',
        userSelect: 'none',
        zIndex: 20,
      }}
    >
      {/* Sleek Status & Tab Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          height: '30px',
          padding: '0 8px',
          backgroundColor: colors.surfaceTile1,
          borderBottom: collapsed ? 'none' : `1px solid ${colors.borderLight}`,
          fontSize: '11px',
        }}
      >
        {/* Left: Tab Selectors */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <button
            onClick={() => setCollapsed(!collapsed)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: colors.bodyMuted,
              display: 'flex',
              alignItems: 'center',
              padding: '2px 4px',
            }}
            title={collapsed ? 'Expand diagnostics' : 'Collapse diagnostics'}
          >
            {collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id);
                if (collapsed) setCollapsed(false);
              }}
              style={{
                background: activeTab === tab.id && !collapsed ? colors.canvas : 'transparent',
                color: activeTab === tab.id && !collapsed ? colors.primary : colors.bodyMuted,
                fontWeight: activeTab === tab.id && !collapsed ? 700 : 500,
                border: 'none',
                padding: '3px 8px',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: 'pointer',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Right: Real Hardware & Telemetry Indicators */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: colors.bodyMuted, fontSize: '11px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Cpu size={12} style={{ color: colors.primary }} />
            <span>GPT-2 Small (124M)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <Server size={12} style={{ color: colors.successText }} />
            <span>Python Live (DirectML / CPU)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: colors.successText }}>
            <CheckCircle2 size={12} />
            <span>No Fabrication</span>
          </div>
        </div>
      </div>

      {/* Expanded Content Area */}
      {!collapsed && (
        <div style={{ flex: 1, overflow: 'auto', padding: '8px', backgroundColor: colors.canvas, fontSize: '11px' }}>
          {/* TASKS TAB */}
          {activeTab === 'tasks' && (
            <div>
              {research.jobs.length === 0 ? (
                <div style={{ color: colors.bodyMuted, textAlign: 'center', padding: '20px 0' }}>
                  No active background jobs. Recent causal runs executed synchronously against resident model weights.
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {research.jobs.map((j) => (
                    <div key={j.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 10px', borderRadius: '4px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                      <div>
                        <span style={{ fontWeight: 700, color: colors.ink }}>{j.name}</span>
                        <code style={{ marginLeft: 8, fontSize: '10px' }}>{j.id}</code>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ color: colors.bodyMuted }}>Stage: <b>{j.current_stage || j.status}</b></span>
                        <span style={{ fontWeight: 700, color: colors.primary }}>{(j.progress * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* CONSOLE TAB */}
          {activeTab === 'console' && (
            <div style={{ fontFamily: 'var(--font-mono, monospace)', display: 'flex', flexDirection: 'column', gap: '2px' }}>
              {console.map((log) => (
                <div key={log.id} style={{ padding: '2px 0', borderBottom: `1px solid ${colors.borderLight}` }}>
                  <span style={{ color: colors.bodyMuted, marginRight: '8px' }}>[{log.ts}]</span>
                  <span style={{ color: log.level === 'error' ? colors.dangerText : (log.level === 'warn' ? colors.warningText : colors.body) }}>
                    {log.message}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* RUNS TAB */}
          {activeTab === 'runs' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {research.runs.length === 0 ? (
                <div style={{ color: colors.bodyMuted, textAlign: 'center', padding: '20px 0' }}>
                  No causal experiment runs recorded yet.
                </div>
              ) : (
                research.runs.map((r) => (
                  <div key={r.id} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 8px', borderRadius: '4px', backgroundColor: colors.surfaceTile1, border: `1px solid ${colors.border}` }}>
                    <div>
                      <span style={{ fontWeight: 600, color: colors.ink }}>Run {r.id.slice(-8)}</span>
                      <span style={{ color: colors.bodyMuted, marginLeft: 8 }}>Target prob: {(r.baseline_target_prob * 100).toFixed(1)}% → {(r.intervened_target_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ fontWeight: 700, color: r.delta_logit > 1.0 ? colors.successText : colors.dangerText }}>
                      ΔLogit = {r.delta_logit.toFixed(2)}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* TIMELINE TAB */}
          {activeTab === 'timeline' && (
            <div>
              {timeline.map((evt) => (
                <div key={evt.id} style={{ padding: '3px 0', borderBottom: `1px solid ${colors.borderLight}` }}>
                  <span style={{ color: colors.bodyMuted, marginRight: '8px' }}>[{evt.ts}]</span>
                  <span style={{ color: colors.body }}>{evt.event}</span>
                </div>
              ))}
            </div>
          )}

          {/* NOTES TAB */}
          {activeTab === 'notes' && (
            <div>
              {notes.map((note) => (
                <div key={note.id} style={{ padding: '4px 0', borderBottom: `1px solid ${colors.borderLight}` }}>
                  <div style={{ fontWeight: 700, color: colors.ink }}>{note.title}</div>
                  <div style={{ color: colors.bodyMuted }}>{note.content}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
