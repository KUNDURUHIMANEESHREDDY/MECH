import React, { useState, useEffect } from 'react';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import { commandManager } from '../../shared/managers/commandManager';

type TabId = 'console' | 'timeline' | 'logs' | 'notes' | 'results' | 'terminal' | 'exports' | 'errors' | 'tasks';

const TABS: { id: TabId; label: string }[] = [
  { id: 'console', label: 'Console' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'logs', label: 'Logs' },
  { id: 'notes', label: 'Notes' },
  { id: 'results', label: 'Results' },
  { id: 'terminal', label: 'Terminal' },
  { id: 'exports', label: 'Exports' },
  { id: 'errors', label: 'Errors' },
  { id: 'tasks', label: 'Tasks' },
];

export const BottomWorkspace: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('console');
  const console = useWorkspaceStore((s) => s.console);
  const timeline = useWorkspaceStore((s) => s.timeline);
  const notes = useWorkspaceStore((s) => s.notes);
  const [localNotes, setLocalNotes] = useState<typeof notes>([]);

  useEffect(() => {
    const unsub1 = commandManager.subscribe('console.log', () => {
      // console state is in the store; force re-render by reading it
      useWorkspaceStore.getState();
    });
    const unsub2 = commandManager.subscribe('timeline.event', () => {
      useWorkspaceStore.getState();
    });
    const unsub3 = commandManager.subscribe('note.created', (e) => {
      setLocalNotes((prev) => [...prev, e.payload.note]);
    });
    return () => { unsub1(); unsub2(); unsub3(); };
  }, []);

  useEffect(() => {
    setLocalNotes(notes);
  }, [notes]);

  return (
    <div className="shell-bottom" style={bottomStyle}>
      <div style={tabBarStyle}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              ...tabStyle,
              ...(activeTab === tab.id ? activeTabStyle : {}),
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
        {activeTab === 'console' && (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
            {console.map((log) => (
              <div key={log.id} style={{ padding: '2px 0', borderBottom: '1px solid var(--border-light)' }}>
                <span style={{ color: 'var(--text-muted)', marginRight: '8px' }}>[{log.ts}]</span>
                <span style={{
                  color: log.level === 'error' ? 'var(--danger)' : log.level === 'warn' ? 'var(--warning)' : 'var(--text)',
                }}>
                  {log.message}
                </span>
              </div>
            ))}
          </div>
        )}
        {activeTab === 'timeline' && (
          <div style={{ fontSize: '12px' }}>
            {timeline.map((evt) => (
              <div key={evt.id} style={{ padding: '4px 0', borderBottom: '1px solid var(--border-light)' }}>
                <span style={{ color: 'var(--text-muted)', marginRight: '8px' }}>[{evt.ts}]</span>
                {evt.event}
              </div>
            ))}
          </div>
        )}
        {activeTab === 'notes' && (
          <div style={{ fontSize: '12px' }}>
            {localNotes.map((note) => (
              <div key={note.id} style={{ padding: '4px 0', borderBottom: '1px solid var(--border-light)' }}>
                <div style={{ fontWeight: 600 }}>{note.title}</div>
                <div style={{ color: 'var(--text-muted)' }}>{note.content}</div>
              </div>
            ))}
          </div>
        )}
        {!['console', 'timeline', 'notes'].includes(activeTab) && (
          <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            {TABS.find((t) => t.id === activeTab)?.label} — coming soon.
          </div>
        )}
      </div>
    </div>
  );
};

const bottomStyle: React.CSSProperties = {
  height: 'var(--console-h, 180px)',
  background: 'var(--bg-elev, #ffffff)',
  borderTop: '1px solid var(--border, #e0e0e0)',
  display: 'flex',
  flexDirection: 'column',
  minHeight: 0,
};

const tabBarStyle: React.CSSProperties = {
  display: 'flex',
  gap: '2px',
  padding: '4px 8px',
  borderBottom: '1px solid var(--border-light, #f0f0f0)',
  background: 'var(--bg-elev-2, #f5f5f5)',
};

const tabStyle: React.CSSProperties = {
  background: 'transparent',
  border: 'none',
  padding: '4px 10px',
  borderRadius: '4px',
  fontSize: '11px',
  cursor: 'pointer',
  color: 'var(--text-muted, #7a7a7a)',
};

const activeTabStyle: React.CSSProperties = {
  background: 'var(--bg-elev, #ffffff)',
  color: 'var(--text, #1d1d1f)',
  fontWeight: 500,
};
