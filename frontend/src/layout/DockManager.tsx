import React from 'react';
import { X } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { panelRegistry } from '../services/panelRegistry';

interface DockManagerProps {
  children: Record<string, React.ReactNode>;
  darkMode: boolean;
}

export const DockManager: React.FC<DockManagerProps> = ({ children }) => {
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

  return (
    <div className="dock-manager">
      <div className="dock-toolbar">
        <span className="dock-toolbar-label">Panels:</span>
        {panels.map(p => {
          const isVisible = !!state.visiblePanels[p.id];
          return (
            <button
              key={p.id}
              onClick={() => togglePanel(p.id)}
              className={'dock-toggle-btn' + (isVisible ? ' active' : '')}
            >
              <span><p.icon size={13} /></span> {p.title}
            </button>
          );
        })}
      </div>

      <div className="dock-grid">
        {Object.entries(children).map(([id, node]) => {
          if (!state.visiblePanels[id]) return null;
          const def = panelRegistry.get(id);
          return (
            <div key={id} className="dock-panel">
              <div className="dock-panel-header">
                <span>{def?.icon ? <def.icon size={13} /> : null} {def?.title ?? id}</span>
                <button onClick={() => togglePanel(id)} className="dock-panel-close"><X size={14} /></button>
              </div>
              <div className="dock-panel-body">{node}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
