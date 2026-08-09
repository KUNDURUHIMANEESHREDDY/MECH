import React, { useState, useEffect } from 'react';
import { Command } from 'cmdk';
import { Search, Flame, Zap, Layers, Type, Brain, Cpu, FileText, Activity } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { commandBus } from '../utils/eventBus';
import './CommandPalette.css';

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ open, onOpenChange }) => {
  const [search, setSearch] = useState('');
  const { pages, activePage, setActivePage, openPanel } = useAppStore();

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

  const entries = Object.entries(pages);

  const handleSelect = (key: string) => {
    setActivePage(key);
    onOpenChange(false);
    commandBus.emit('page:changed', { page: key });
  };

  return (
    <div className="cmd-overlay" onClick={() => onOpenChange(false)}>
      <div className="cmd-modal" onClick={(e) => e.stopPropagation()}>
        <Command label="Global Command Palette">
          <div className="cmd-input-row">
            <Search size={16} className="cmd-search-icon" />
            <Command.Input
              value={search}
              onValueChange={setSearch}
              placeholder="Search pages, panels, or commands..."
              className="cmd-input"
            />
          </div>

          <Command.List className="cmd-list">
            <Command.Empty className="cmd-empty">
              No results found for "{search}".
            </Command.Empty>

            <Command.Group heading="Pages" className="cmd-group">
              {entries
                .filter(([, page]) => page.label.toLowerCase().includes(search.toLowerCase()))
                .map(([key, page]) => (
                  <Command.Item
                    key={key}
                    value={key}
                    onSelect={() => handleSelect(key)}
                    className={`cmd-item ${activePage === key ? 'active' : ''}`}
                  >
                    {page.icon && <page.icon size={16} />}
                    <span>{page.label}</span>
                    {activePage === key && <span className="cmd-badge">Active</span>}
                  </Command.Item>
                ))}
            </Command.Group>
          </Command.List>
        </Command>
      </div>
    </div>
  );
};
