# MECH Platform — IDE Design System

## Layout (VS Code-style IDE)

```
┌──────────┬───────────────────────────┬───────────────────────┐
│ Activity │  Sidebar (260px)          │  Main Content (flex)  │
│ Bar      │  - File tree              │  - Graphs / Heatmaps  │
│ (52px)   │  - Explorer               │  - Activation Viewer  │
│          │  - Model hierarchy         │  - Prompt + Run      │
│          │  - Experiments             │  - Dock panels       │
├──────────┴───────────────────────────┴───────────────────────┤
│  Console (collapsible, 180px)  —  Logs / Build output        │
├──────────────────────────────────────────────────────────────┤
│  StatusBar (26px)                                            │
└──────────────────────────────────────────────────────────────┘
```

## Tokens

```css
--bg: #1e1e2e;
--bg-sidebar: #181825;
--bg-activity: #11111b;
--bg-elev: #252536;
--bg-elev-2: #2e2e42;
--bg-hover: #313244;
--bg-active: #45475a;
--border: #2a2a3c;
--border-light: #3a3a50;
--text: #cdd6f4;
--text-dim: #a6adc8;
--text-muted: #6c7086;
--accent: #89b4fa;
--accent-hover: #74c7ec;
--accent-soft: rgba(137, 180, 250, 0.1);
--danger: #f38ba8;
--success: #a6e3a1;
--warning: #f9e2af;
--radius: 6px;
--radius-lg: 10px;
--font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
--font-mono: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
```

### Activity Bar (52px)
### Sidebar (260px)
### Main Content (flex-1)
### Console (180px, collapsible)
### StatusBar (26px)

---

## AI Research Assistant Integration (Canvas & Studio)

1. **Collaborative Canvas (ReactFlow)**:
   * When the AI Research Assistant executes an experiment or runs circuit discovery (ACDC), candidate circuits and causal mediation flows are rendered automatically onto the visual graph canvas.
   * Nodes represent Attention Heads, Neurons, and SAE Features with live attribution weights and mediation scores.

2. **Interactive Research Notebooks**:
   * Synchronized notebook environment where the AI assistant writes experimental rationale, logs raw tensor measurements, and embeds interactive heatmaps.
   * Allows human researchers to modify prompts, tweak intervention parameters, and re-trigger execution in real time.

3. **Command Palette & AI Chat Drawer**:
   * Quick-launch mechanistic queries via `Ctrl/Cmd + K` or through the interactive research co-pilot drawer.
