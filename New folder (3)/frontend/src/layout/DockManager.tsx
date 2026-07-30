import React from 'react';
import { useAppStore } from '../store/useAppStore';
import { panelRegistry } from '../services/panelRegistry';

interface DockManagerProps {
  children: Record<string, React.ReactNode>;
  darkMode: boolean;
}

export const DockManager: React.FC<DockManagerProps> = ({ children, darkMode }) => {
  const [state, setState] = useAppStore();

  const togglePanel = (id: string) => {
    setState(prev => ({
      visiblePanels: {
        ...prev.visiblePanels,
        [id]: !prev.visiblePanels[id],
      },
    }));
  };

  const panels = panelRegistry.list();
  const borderColor = darkMode ? '#2a2a3a' : '#d0d0d0';
  const panelBg = darkMode ? '#16161e' : '#ffffff';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', overflow: 'hidden' }}>
      {/* Panel Toggle Toolbar */}
      <div style={{ display: 'flex', gap: 6, padding: '6px 12px', background: panelBg, borderBottom: `1px solid ${borderColor}`, alignItems: 'center' }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: '#888', textTransform: 'uppercase', marginRight: 8 }}>Panels:</span>
        {panels.map(p => {
          const isVisible = !!state.visiblePanels[p.id];
          return (
            <button
              key={p.id}
              onClick={() => togglePanel(p.id)}
              style={{
                background: isVisible ? '#3b82f6' : (darkMode ? '#2a2a3a' : '#e0e0e0'),
                color: isVisible ? '#fff' : (darkMode ? '#aaa' : '#444'),
                border: 'none',
                borderRadius: 4,
                padding: '4px 8px',
                fontSize: 11,
                cursor: 'pointer',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <span>{p.icon}</span> {p.title}
            </button>
          );
        })}
      </div>

      {/* Main Grid View */}
      <div style={{ flex: 1, display: 'flex', gap: 12, padding: 12, overflow: 'auto' }}>
        {Object.entries(children).map(([id, node]) => {
          if (!state.visiblePanels[id]) return null;
          const def = panelRegistry.get(id);
          return (
            <div
              key={id}
              style={{
                flex: 1,
                minWidth: 320,
                background: panelBg,
                borderRadius: 8,
                padding: 12,
                border: `1px solid ${borderColor}`,
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <div style={{ fontSize: 11, fontWeight: 700, color: '#888', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8, display: 'flex', justifyContent: 'space-between' }}>
                <span>{def?.icon} {def?.title ?? id}</span>
                <button
                  onClick={() => togglePanel(id)}
                  style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer', fontSize: 12 }}
                >
                  ✕
                </button>
              </div>
              <div style={{ flex: 1, overflow: 'auto' }}>{node}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
