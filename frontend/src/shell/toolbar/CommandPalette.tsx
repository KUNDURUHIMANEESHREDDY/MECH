import React, { useState, useEffect } from 'react';
import { Command } from 'cmdk';
import { Search, Flame, Zap, Layers, Type, Brain, Cpu, FileText, Activity } from 'lucide-react';
import { useWorkspaceStore } from '../../shared/stores/workspace';
import { pluginRegistry } from '../../panel-system/pluginRegistry';

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ open, onOpenChange }) => {
  const [search, setSearch] = useState('');
  const { openPanel, openResource } = useWorkspaceStore();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        onOpenChange(!open);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onOpenChange]);

  if (!open) return null;

  const plugins = pluginRegistry.list();

  const handleSelectPanel = (panelId: string) => {
    openPanel(panelId);
    onOpenChange(false);
  };

  return (
    <div
      onClick={() => onOpenChange(false)}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.3)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '15vh',
        zIndex: 1000,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '540px',
          background: 'var(--color-canvas, #ffffff)',
          border: '1px solid var(--border, #e0e0e0)',
          borderRadius: '12px',
          boxShadow: 'var(--shadow-lg, 0 10px 40px rgba(0,0,0,0.15))',
          overflow: 'hidden',
        }}
      >
        <Command label="Global Command Palette">
          <div style={{ display: 'flex', alignItems: 'center', padding: '0 14px', borderBottom: '1px solid var(--border-light)' }}>
            <Search size={16} style={{ color: 'var(--text-muted)', marginRight: '10px' }} />
            <Command.Input
              value={search}
              onValueChange={setSearch}
              placeholder="Search resources, panels, or commands (e.g. 'attention')..."
              style={{
                width: '100%',
                height: '48px',
                border: 'none',
                outline: 'none',
                fontSize: '14px',
                color: 'var(--text, #1d1d1f)',
                background: 'transparent',
              }}
            />
          </div>

          <Command.List style={{ maxHeight: '320px', overflowY: 'auto', padding: '8px' }}>
            <Command.Empty style={{ padding: '16px', textAlign: 'center', fontSize: '13px', color: 'var(--text-muted)' }}>
              No results found for "{search}".
            </Command.Empty>

            <Command.Group heading="Panel Plugins" style={{ fontSize: '11px', color: 'var(--text-muted)', padding: '4px 8px' }}>
              {plugins.map((plugin) => (
                <Command.Item
                  key={plugin.id}
                  onSelect={() => handleSelectPanel(plugin.id)}
                  style={itemStyle}
                >
                  <Flame size={14} style={{ color: 'var(--color-primary)' }} />
                  <span style={{ fontWeight: 500 }}>{plugin.title}</span>
                  <span style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-muted)' }}>Panel</span>
                </Command.Item>
              ))}
            </Command.Group>
          </Command.List>
        </Command>
      </div>
    </div>
  );
};

const itemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '10px',
  padding: '8px 12px',
  borderRadius: '6px',
  fontSize: '13px',
  color: 'var(--text, #1d1d1f)',
  cursor: 'pointer',
  userSelect: 'none',
};
