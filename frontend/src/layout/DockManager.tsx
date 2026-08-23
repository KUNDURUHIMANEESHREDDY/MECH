import React, { useState, useEffect, useRef } from 'react';
import { useWorkspaceStore } from '../shared/stores/workspace';
import { pluginRegistry } from '../panel-system/pluginRegistry';
import { colors } from '../design/tokens/colors';
import {
  Maximize2,
  Minimize2,
  X,
  LayoutGrid,
  Columns,
  Sparkles,
  Plus,
} from 'lucide-react';

const EmptyState: React.FC = () => (
  <div
    style={{
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: colors.bodyMuted,
      gap: '16px',
      fontSize: '14px',
      userSelect: 'none',
      padding: '40px',
      backgroundColor: colors.canvas,
      height: '100%',
      boxSizing: 'border-box',
    }}
  >
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontWeight: 700, fontSize: '16px', marginBottom: '6px', color: colors.ink }}>
        Scientific Research Canvas
      </div>
      <div style={{ fontSize: '13px', color: colors.bodyMuted, maxWidth: '340px', lineHeight: 1.5 }}>
        Select a tool from the Navigator or press <kbd style={{ background: colors.surfacePearl, border: `1px solid ${colors.border}`, borderRadius: '4px', padding: '1px 5px', fontSize: '11px', fontWeight: 600 }}>⌘K</kbd> to open an analysis view.
      </div>
    </div>
  </div>
);

export const DockManager: React.FC = () => {
  const visiblePanels = useWorkspaceStore((s) => s.visiblePanels);
  const closePanel = useWorkspaceStore((s) => s.closePanel);
  const openPanel = useWorkspaceStore((s) => s.openPanel);

  const [layoutMode, setLayoutMode] = useState<'tabs' | 'split' | 'grid'>('tabs');
  const [activeTabId, setActiveTabId] = useState<string | null>(null);

  const openPanelIds = Object.entries(visiblePanels)
    .filter(([, visible]) => visible)
    .map(([id]) => id);

  // Track panel additions so we can focus a panel the researcher actually
  // opened (drawer click or hash route) without hijacking the default tab
  // on first mount.
  const prevOpenRef = useRef<string[]>(openPanelIds);
  useEffect(() => {
    const prev = prevOpenRef.current;
    const newlyOpened = openPanelIds.filter((id) => !prev.includes(id));
    if (newlyOpened.length > 0) {
      setActiveTabId(newlyOpened[newlyOpened.length - 1]);
    }
    prevOpenRef.current = openPanelIds;
  }, [openPanelIds.join('|')]);

  // Set default active tab if current active tab is closed or null
  const currentTabId = (activeTabId && openPanelIds.includes(activeTabId))
    ? activeTabId
    : (openPanelIds.length > 0 ? openPanelIds[0] : null);

  if (openPanelIds.length === 0) {
    return <EmptyState />;
  }

  const dummyCtx: any = {
    selection: { resource: null, layer: null, head: null, neuron: null, token: null },
    workspace: {},
  };

  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: colors.canvas,
        overflow: 'hidden',
        boxSizing: 'border-box',
      }}
    >
      {/* Top Canvas Tab & Layout Switcher Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          backgroundColor: colors.surfaceTile1,
          borderBottom: `1px solid ${colors.border}`,
          padding: '0 8px',
          height: '36px',
          userSelect: 'none',
          boxSizing: 'border-box',
        }}
      >
        {/* Open Panels Tabs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '2px', overflowX: 'auto', flex: 1, height: '100%' }}>
          {openPanelIds.map((panelId) => {
            const plugin = pluginRegistry.get(panelId);
            const title = plugin?.title || panelId;
            const isActive = currentTabId === panelId;

            return (
              <div
                key={panelId}
                onClick={() => setActiveTabId(panelId)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '0 12px',
                  height: '100%',
                  fontSize: '12px',
                  fontWeight: isActive ? 700 : 500,
                  color: isActive ? colors.primary : colors.body,
                  backgroundColor: isActive ? colors.canvas : 'transparent',
                  borderRight: `1px solid ${colors.borderLight}`,
                  borderBottom: isActive ? `2px solid ${colors.primary}` : 'none',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                <span>{title}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    closePanel(panelId);
                    if (currentTabId === panelId) {
                      const next = openPanelIds.find((id) => id !== panelId);
                      setActiveTabId(next || null);
                    }
                  }}
                  style={{
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    color: colors.bodyMuted,
                    fontSize: '14px',
                    padding: 0,
                    lineHeight: 1,
                    display: 'flex',
                    alignItems: 'center',
                  }}
                  title="Close view"
                >
                  <X size={12} />
                </button>
              </div>
            );
          })}
        </div>

        {/* Layout Mode Toggles */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginLeft: '8px' }}>
          <button
            onClick={() => setLayoutMode('tabs')}
            style={{
              ...modeBtnStyle,
              backgroundColor: layoutMode === 'tabs' ? colors.primary : colors.canvas,
              color: layoutMode === 'tabs' ? colors.onPrimary : colors.body,
            }}
            title="Tabbed Single View (Full Width)"
          >
            Tabbed
          </button>
          <button
            onClick={() => setLayoutMode('split')}
            style={{
              ...modeBtnStyle,
              backgroundColor: layoutMode === 'split' ? colors.primary : colors.canvas,
              color: layoutMode === 'split' ? colors.onPrimary : colors.body,
            }}
            title="Split Side-by-Side View"
          >
            <Columns size={12} /> Split
          </button>
          <button
            onClick={() => setLayoutMode('grid')}
            style={{
              ...modeBtnStyle,
              backgroundColor: layoutMode === 'grid' ? colors.primary : colors.canvas,
              color: layoutMode === 'grid' ? colors.onPrimary : colors.body,
            }}
            title="Grid Multi-Panel View"
          >
            <LayoutGrid size={12} /> Grid
          </button>
        </div>
      </div>

      {/* Canvas View Content */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex' }}>
        {layoutMode === 'tabs' && currentTabId && (
          <div style={{ flex: 1, height: '100%', overflow: 'auto' }}>
            {(() => {
              const plugin = pluginRegistry.get(currentTabId);
              if (!plugin) return <EmptyState />;
              return <plugin.Body {...dummyCtx} />;
            })()}
          </div>
        )}

        {layoutMode === 'split' && (
          <div style={{ flex: 1, display: 'grid', gridTemplateColumns: `repeat(${Math.min(openPanelIds.length, 3)}, 1fr)`, height: '100%', overflow: 'hidden' }}>
            {openPanelIds.slice(0, 3).map((panelId) => {
              const plugin = pluginRegistry.get(panelId);
              if (!plugin) return null;
              return (
                <div key={panelId} style={{ borderRight: `1px solid ${colors.border}`, height: '100%', overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
                  <div style={{ padding: '6px 12px', fontSize: '11px', fontWeight: 700, backgroundColor: colors.surfaceTile1, borderBottom: `1px solid ${colors.borderLight}` }}>
                    {plugin.title}
                  </div>
                  <div style={{ flex: 1, overflow: 'auto' }}>
                    <plugin.Body {...dummyCtx} />
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {layoutMode === 'grid' && (
          <div style={{ flex: 1, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '8px', padding: '8px', overflow: 'auto' }}>
            {openPanelIds.map((panelId) => {
              const plugin = pluginRegistry.get(panelId);
              if (!plugin) return null;
              return (
                <div key={panelId} style={{ border: `1px solid ${colors.border}`, borderRadius: '8px', backgroundColor: colors.surfaceTile1, overflow: 'hidden', height: '480px', display: 'flex', flexDirection: 'column' }}>
                  <div style={{ padding: '6px 12px', fontSize: '11px', fontWeight: 700, backgroundColor: colors.canvas, borderBottom: `1px solid ${colors.borderLight}` }}>
                    {plugin.title}
                  </div>
                  <div style={{ flex: 1, overflow: 'auto' }}>
                    <plugin.Body {...dummyCtx} />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

const modeBtnStyle: React.CSSProperties = {
  padding: '3px 8px',
  borderRadius: '4px',
  border: `1px solid ${colors.border}`,
  fontSize: '11px',
  fontWeight: 600,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  gap: '4px',
};
