import React from 'react';
import { useSelectionStore } from '../../shared/stores/selection';

export const Inspector: React.FC = () => {
  const selection = useSelectionStore();

  return (
    <aside className="shell-inspector" style={inspectorStyle}>
      <div style={headerStyle}>
        <span style={{ fontWeight: 600, fontSize: '12px', letterSpacing: '-0.2px' }}>Inspector</span>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
        {selection.resource ? (
          <div>
            <div style={{ fontWeight: 600, marginBottom: '8px' }}>{selection.resource.label}</div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>
              Kind: {selection.resource.kind}
            </div>
            {selection.layer !== null && (
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>
                Layer: {selection.layer}
              </div>
            )}
            {selection.head !== null && (
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>
                Head: {selection.head}
              </div>
            )}
            {selection.neuron !== null && (
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px' }}>
                Neuron: {selection.neuron}
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
            Select a resource to inspect.
          </div>
        )}
      </div>
    </aside>
  );
};

const inspectorStyle: React.CSSProperties = {
  width: 'var(--sidebar-width, 240px)',
  background: 'var(--bg-sidebar, #fafafa)',
  borderLeft: '1px solid var(--border, #e0e0e0)',
  display: 'flex',
  flexDirection: 'column',
  userSelect: 'none',
  fontSize: '12px',
};

const headerStyle: React.CSSProperties = {
  height: '36px',
  display: 'flex',
  alignItems: 'center',
  padding: '0 12px',
  borderBottom: '1px solid var(--border-light, #f0f0f0)',
  color: 'var(--text, #1d1d1f)',
};
