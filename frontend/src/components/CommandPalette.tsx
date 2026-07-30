import React, { useState, useEffect } from 'react';
import { useAppStore } from '../store/useAppStore';

interface CommandPaletteProps {
  onLoadModel: (name: string) => void;
  onRunPrompt: () => void;
  darkMode: boolean;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ onLoadModel, onRunPrompt, darkMode }) => {
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
    { label: '⚡ Load GPT-2 Model', action: () => onLoadModel('gpt2') },
    { label: '▶ Run Current Prompt', action: () => onRunPrompt() },
    { label: '🌓 Toggle Dark Mode', action: () => setState(s => ({ darkMode: !s.darkMode })) },
    { label: '🔤 Toggle Token Viewer', action: () => setState(s => ({ visiblePanels: { ...s.visiblePanels, token_viewer: !s.visiblePanels.token_viewer } })) },
    { label: '🔥 Toggle Attention Heatmap', action: () => setState(s => ({ visiblePanels: { ...s.visiblePanels, attention_heatmap: !s.visiblePanels.attention_heatmap } })) },
    { label: '🔮 Toggle Prediction Inspector', action: () => setState(s => ({ visiblePanels: { ...s.visiblePanels, prediction_inspector: !s.visiblePanels.prediction_inspector } })) },
    { label: '🥞 Toggle Layer Inspector', action: () => setState(s => ({ visiblePanels: { ...s.visiblePanels, layer_inspector: !s.visiblePanels.layer_inspector } })) },
  ];

  const filtered = commands.filter(c => c.label.toLowerCase().includes(query.toLowerCase()));

  const bg = darkMode ? '#1e1e2e' : '#ffffff';
  const fg = darkMode ? '#ffffff' : '#111111';
  const itemBg = darkMode ? '#2a2a3c' : '#f0f0f5';
  const border = darkMode ? '#3a3a4c' : '#cccccc';

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999 }}>
      <div style={{ background: bg, color: fg, borderRadius: 12, border: `1px solid ${border}`, width: 500, maxWidth: '90vw', padding: 16, boxShadow: '0 20px 40px rgba(0,0,0,0.5)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <span style={{ fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, color: '#888' }}>Command Palette (Ctrl+Shift+P)</span>
          <button onClick={() => setState({ commandPaletteOpen: false })} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer' }}>✕</button>
        </div>
        <input
          autoFocus
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Type a command or search..."
          style={{ width: '100%', padding: '10px 14px', borderRadius: 6, border: `1px solid ${border}`, background: darkMode ? '#12121a' : '#fff', color: fg, fontSize: 14, outline: 'none', marginBottom: 12 }}
        />
        <div style={{ maxHeight: 260, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {filtered.length === 0 ? (
            <div style={{ color: '#888', padding: 12, textAlign: 'center', fontSize: 13 }}>No matching commands</div>
          ) : (
            filtered.map((cmd, idx) => (
              <button
                key={idx}
                onClick={() => { cmd.action(); setState({ commandPaletteOpen: false }); }}
                style={{ background: itemBg, color: fg, border: 'none', borderRadius: 6, padding: '10px 14px', textAlign: 'left', cursor: 'pointer', fontSize: 13, fontWeight: 600, display: 'flex', justifyContent: 'space-between' }}
              >
                <span>{cmd.label}</span>
                <span style={{ fontSize: 11, color: '#888' }}>↵ Run</span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
