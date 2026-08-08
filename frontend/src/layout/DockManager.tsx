import React from 'react';
import { useWorkspaceStore } from '../shared/stores/workspace';
import { pluginRegistry } from '../panel-system/pluginRegistry';

const EmptyState: React.FC = () => (
  <div
    style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'var(--text-muted, #7a7a7a)',
      gap: '16px',
      fontSize: '14px',
      userSelect: 'none',
      padding: '40px',
    }}
  >
    <svg
      width="52"
      height="52"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.25"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ opacity: 0.3 }}
    >
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M9 3v18" />
      <path d="M3 9h6" />
      <path d="M3 15h6" />
    </svg>
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontWeight: 600, fontSize: '15px', marginBottom: '6px', color: 'var(--text, #1d1d1f)' }}>
        Visual Research Canvas
      </div>
      <div style={{ fontSize: '13px', color: 'var(--text-muted, #7a7a7a)', maxWidth: '280px', lineHeight: 1.5 }}>
        Double-click a resource in the Navigator or press <kbd style={{ background: 'var(--bg-elev-2, #f0f0f0)', border: '1px solid var(--border, #d0d0d0)', borderRadius: '4px', padding: '1px 5px', fontSize: '11px', fontWeight: 600 }}>⌘K</kbd> to open a panel.
      </div>
    </div>
  </div>
);

export const DockManager: React.FC = () => {
  const visiblePanels = useWorkspaceStore((s) => s.visiblePanels);
  const closePanel = useWorkspaceStore((s) => s.closePanel);

  const openPanelIds = Object.entries(visiblePanels)
    .filter(([, visible]) => visible)
    .map(([id]) => id);

  if (openPanelIds.length === 0) {
    return <EmptyState />;
  }

  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexWrap: 'wrap',
        gap: '8px',
        padding: '12px',
        overflow: 'auto',
        alignContent: 'flex-start',
      }}
    >
      {openPanelIds.map((panelId) => {
        const plugin = pluginRegistry.get(panelId);
        if (!plugin) return null;

        const dummyCtx: any = {
          selection: { resource: null, layer: null, head: null, neuron: null, token: null },
          workspace: {},
        };

        return (
          <div
            key={panelId}
            style={{
              background: 'var(--color-canvas, #ffffff)',
              border: '1px solid var(--border, #e0e0e0)',
              borderRadius: '10px',
              minWidth: '320px',
              flex: '1 1 320px',
              maxWidth: '600px',
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
              boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 12px',
                borderBottom: '1px solid var(--border-light, #f0f0f0)',
                background: 'var(--bg-elev-1, #fafafa)',
                fontSize: '12px',
                fontWeight: 600,
                color: 'var(--text, #1d1d1f)',
              }}
            >
              <span>{plugin.title}</span>
              <button
                onClick={() => closePanel(panelId)}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-muted)',
                  padding: '2px 4px',
                  borderRadius: '4px',
                  fontSize: '14px',
                  lineHeight: 1,
                }}
                title="Close panel"
              >
                ×
              </button>
            </div>
            <div style={{ flex: 1, overflow: 'auto', padding: '12px' }}>
              <plugin.Body {...dummyCtx} />
            </div>
          </div>
        );
      })}
    </div>
  );
};
