import React, { useState, useEffect } from 'react';

export default function CommandPalette({ isOpen, onClose, onExecuteCommand }) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);

  const COMMANDS = [
    { id: 'load-model', category: 'Model', label: 'Load Model into Runtime', action: () => onExecuteCommand('load-model') },
    { id: 'run-prompt', category: 'Debugger', label: 'Run Prompt & Analyze Activations', action: () => onExecuteCommand('run-prompt') },
    { id: 'open-experiment', category: 'Experiments', label: 'Open Circuit Intervention Matrix', action: () => onExecuteCommand('open-experiments') },
    { id: 'search-neuron', category: 'Analysis', label: 'Search Neuron Activation (Layer/Index)', action: () => onExecuteCommand('search-neuron') },
    { id: 'search-feature', category: 'SAE', label: 'Search Sparse Autoencoder Feature', action: () => onExecuteCommand('search-feature') },
    { id: 'open-sessions', category: 'Sessions', label: 'Open Recent Session Explorer', action: () => onExecuteCommand('open-sessions') },
    { id: 'toggle-theme', category: 'Settings', label: 'Toggle Application Theme', action: () => onExecuteCommand('toggle-theme') },
    { id: 'toggle-dock', category: 'View', label: 'Toggle Dock & Research Panels', action: () => onExecuteCommand('toggle-dock') }
  ];

  const filtered = COMMANDS.filter((cmd) =>
    cmd.label.toLowerCase().includes(query.toLowerCase()) ||
    cmd.category.toLowerCase().includes(query.toLowerCase())
  );

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isOpen) return;
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % (filtered.length || 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev - 1 + filtered.length) % (filtered.length || 1));
      } else if (e.key === 'Enter' && filtered[selectedIndex]) {
        e.preventDefault();
        filtered[selectedIndex].action();
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filtered, selectedIndex, onClose]);

  if (!isOpen) return null;

  return (
    <div className="command-palette-backdrop" onClick={onClose} data-testid="command-palette">
      <div className="command-palette-modal" onClick={(e) => e.stopPropagation()}>
        <div className="command-palette-input-wrapper">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            className="command-palette-input"
            placeholder="Type a command or search (e.g. Load Model, Search Neuron)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
          <span className="kbd-shortcut">ESC to exit</span>
        </div>
        <div className="command-palette-results">
          {filtered.length === 0 ? (
            <div className="command-no-results">No matching commands found</div>
          ) : (
            filtered.map((cmd, idx) => (
              <div
                key={cmd.id}
                className={`command-item ${idx === selectedIndex ? 'selected' : ''}`}
                onClick={() => {
                  cmd.action();
                  onClose();
                }}
              >
                <span className="command-category">{cmd.category}</span>
                <span className="command-label">{cmd.label}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
