import React, { useState, useEffect } from 'react';
import { CornerDownLeft, X } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

interface CommandPaletteProps {
  onLoadModel: (name: string) => void;
  onRunPrompt: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ onLoadModel, onRunPrompt }) => {
  const [state, setState] = useAppStore();
  const [query, setQuery] = useState('');

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'p') {
        e.preventDefault();
        setState(prev => ({ commandPaletteOpen: !prev.commandPaletteOpen }));
      }
      if (e.key === 'Escape' && state.commandPaletteOpen) {
        setState({ commandPaletteOpen: false });
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state.commandPaletteOpen]);

  if (!state.commandPaletteOpen) return null;

  const commands = [
    { label: 'Load GPT-2 Model', action: () => onLoadModel('gpt2') },
    { label: 'Run Current Prompt', action: () => onRunPrompt() },
    { label: 'Toggle Token Viewer', action: () => setState(s => ({ visiblePanels: { ...s.visiblePanels, token_viewer: !s.visiblePanels.token_viewer } })) },
    { label: 'Toggle Attention Heatmap', action: () => setState(s => ({ visiblePanels: { ...s.visiblePanels, attention_heatmap: !s.visiblePanels.attention_heatmap } })) },
    { label: 'Toggle Prediction Inspector', action: () => setState(s => ({ visiblePanels: { ...s.visiblePanels, prediction_inspector: !s.visiblePanels.prediction_inspector } })) },
    { label: 'Toggle Layer Inspector', action: () => setState(s => ({ visiblePanels: { ...s.visiblePanels, layer_inspector: !s.visiblePanels.layer_inspector } })) },
  ];

  const filtered = commands.filter(c => c.label.toLowerCase().includes(query.toLowerCase()));

  return (
    <div className="command-palette-backdrop" onClick={() => setState({ commandPaletteOpen: false })}>
      <div className="command-palette-modal" onClick={e => e.stopPropagation()}>
        <div className="command-palette-header">
          <span className="command-palette-title">Command Palette</span>
          <button onClick={() => setState({ commandPaletteOpen: false })} className="cmd-close-btn"><X size={14} /></button>
        </div>
        <div className="command-palette-input-wrapper">
          <input
            autoFocus
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Type a command or search..."
            className="command-palette-input"
          />
        </div>
        <div className="command-palette-results">
          {filtered.length === 0 ? (
            <div className="cmd-empty">No matching commands</div>
          ) : (
            filtered.map((cmd, idx) => (
              <button
                key={idx}
                onClick={() => { cmd.action(); setState({ commandPaletteOpen: false }); }}
                className="command-item"
              >
                <span>{cmd.label}</span>
                <span className="kbd-shortcut"><CornerDownLeft size={10} style={{ verticalAlign: 'middle', marginRight: 2 }} />Run</span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
