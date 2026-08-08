---
slug: redesign-ui-shell
status: awaiting-approval
intent: unclear
review_required: true
plan_path: .omo/plans/redesign-ui-shell.md
plan_sha256: null
review_round_id: null
pending-action: write and review .omo/plans/redesign-ui-shell.md
review:
  momus:
    status: pending
    workspace_root: null
    runtime_home: null
    target: .omo/plans/redesign-ui-shell.md
    round_id: null
    plan_sha256: null
    launch_id: null
    session: null
    result: null
  independent:
    status: pending
    workspace_root: null
    runtime_home: null
    target: .omo/plans/redesign-ui-shell.md
    round_id: null
    plan_sha256: null
    launch_id: null
    session: null
    result: null
approach: DECISION-MADE (React). User approved React per their explicit stack (Electron/React/TS/Tailwind+shadcn/motion/Zustand/TanStack/FlexLayout/ReactFlow/cmdk). Verified: the LIVE entry is the Vue shell (index.html:10 -> /src/main.ts -> App.vue), but vite.config.mts already wires BOTH vue() and react() plugins, and the feature-complete React tree (App.tsx 19.4KB, DockManager, LayoutManager, zustand store, design tokens, 700-line pages) exists unmounted. Plan = flip index.html to /src/main.tsx, build the Research Canvas shell in React, retire the Vue shell, re-point Playwright. v3 refinements (Changes 1-10: no pages, Finder-like hierarchical Navigator, resource model, panel plugin contract, Panel Registry, Command Bus, workspace-as-state, Spotlight tool palette, multiple workspaces, backend-never-knows-UI; plus manager decomposition and new folder structure; local-only) folded into assumptions A4/A5/A12/A14/A15, components C6b/C8b/C8c/C8d/C11, decisions D7/D8.
---

# Draft: redesign-ui-shell

## Components (topology ledger)
| id | outcome | status | evidence path |
|----|---------|--------|----------------|
| C1 token-system | Apple-light single token layer; all existing var(--ink/--primary/--hairline/--canvas) references resolve; no gradients/glass/neon | active | frontend/src/styles.css:5-30 + 1735-1792 (two conflicting :root); main.ts imports only styles.css; .vue var usage: --ink 182x, --ink-muted 129x, --canvas 21x, --primary 20x, --hairline 4x |
| C2 shell | 5-region Research Canvas shell in React: Toolbar / Navigator / Workspace / Inspector / Bottom Workspace (replaces 4-pane IDE) | active | frontend/src/App.tsx (19.4KB, mounts via main.tsx); Vue Sidebar/Topbar/ActivityBar/StatusBar to retire |
| C3 workspace | Infinite canvas + PanelManager + WindowManager + DockManager: open/close/drag/resize/dock/split/pin/detach/float, pan/zoom/snap, save+restore layouts | active | NO live infra exists (DockManager.tsx + LayoutManager.ts are React-only, unmounted) |
| C4 inspector | Context-sensitive right Inspector via shared selection store | active | new; React NeuronPanel.tsx/LayerSidebar.tsx unmounted |
| C5 bottom | Tabbed bottom workspace: Console/Timeline/Logs/Notes/Results/Terminal/Exports/Errors/Tasks | active | StatusBar.vue:1-23 minimal today |
| C6 sessions | Session restore: window positions, panels, camera/zoom, model, inspector, notes, timeline, console | active | electron/storage.js better-sqlite3 KV SettingsStore; React LayoutManager.ts (localStorage) is only existing pattern, unmounted |
| C6b multi-workspace | Multiple named workspaces (A/B/C), instant switch, layouts preserved per workspace | active | user v3 Change 9 |
| C7 panels-content | The 14 spec panel types (Attention Map, Neuron Explorer, Circuit Viewer, Residual Stream, Logit Lens, ...) mount REAL content where it exists | active | React tree has feature-complete pages (NeuralExplorerView.jsx 738L, Gpt2NeuronExplorer.tsx 701L, TransformerVisualizer.tsx 740L, ReasoningTraceView.jsx 562L); Vue has only 4 real pages (Gpt2View 206L, TransformerExplorer 242L, NeuralExplorerView 627L, + tiny ones) |
| C8 stub-pages | 20+ seven-line Vue stub pages replaced with resource-backed empty-state panels | active | ReportsView.vue:1-9, Logging.vue:1-7, ... |
| C8b registry | Panel Registry: register a plugin (attention/circuit/neuron/embedding/logit-lens/sae/future), NOT hardcoded panel types; adding analysis = registration, never shell edits | active | user v3 Change 5 |
| C8c panel-contract | Panel plugin contract: Header/Toolbar/Body/Inspector/Context Menu/Commands/Persistence — every panel implements it | active | user v3 Change 4 |
| C8d command-bus | Event bus for panel coupling (NeuronSelected -> Inspector/Timeline/Circuit/Console/Notes); panels publish/subscribe, never call each other | active | user v3 Change 6 |
| C9 tests | Vitest stays green (React tests); Playwright shell-smoke green (nav testids + header crumb labels) | active | tests/playwright/shell-smoke.spec.js:1-80 (comment: "Labels must match App.tsx PAGES" but testids live in Sidebar.vue:37); vitest.config.js:8 |
| C10 react-tree | SWITCH the live renderer: index.html -> /src/main.tsx (React) so App.tsx mounts; retire Vue shell (App.vue, main.ts, *.vue components, pinia app.ts) after Playwright re-point; DO NOT mix trees | active | frontend/index.html:10 -> /src/main.ts (Vue); src/main.tsx exists (ReactDOM.createRoot -> App, imports ./styles.css); vite.config.mts:9 plugins: [vue(), react(), vueJsx(), tailwindcss()] — both enabled |
| C11 tool-palette | Spotlight-style tool palette (⌘K / search): type "attention" -> open panel; replaces menu-driven navigation | active | user v3 Change 8 |

## Open assumptions (announced defaults)
| # | assumption | adopted default | rationale | reversible? |
|---|-----------|-----------------|-----------|-------------|
| A1 Theme | LIGHT Apple-pro: #f5f5f7 chrome, white cards, thin borders, large radius (12-16px), soft elevation, generous whitespace; NO gradients/glass/neon/glow | user v2 spec "Visual Style" | yes |
| A2 Typography | System/SF-style stack, larger type, generous whitespace; mono for data values | v2 spec | yes |
| A3 Toolbar | Global-only: logo/search/model/session/Run/Export/Workspace/Settings/Profile; no breadcrumbs/nested menus | v2 spec | yes |
| A4 Navigator | Hierarchical resource tree (Finder-like), NOT flat pages: Workspace/Research/Sessions, Models (GPT2/Gemma/Llama), Assets (Datasets/Exports/Layouts), Bookmarks; double-click a resource -> opens its panel; nothing navigates away | user v3 Change 2/3 | yes |
| A5 Workspace | Infinite canvas: pan/zoom/snap/dock/floating/multi-monitor/save+restore layouts; PERMANENT — nothing navigates away; MULTIPLE workspaces (GPT-2/Gemma/Llama) with instant switch | user v3 Change 1/9 | yes |
| A6 Inspector | Context-sensitive right, auto-updates on selection | v2 spec | yes |
| A7 Bottom workspace | Tabbed console/timeline/logs/notes/results/terminal/exports/errors/tasks | v2 spec | yes |
| A8 Animations | Everything animated; motion/spring; open 0.97->1.0, inspector slide, dock preview+bounce, spotlight search; no spinners (skeletons/progress) | v2 spec + stack msg | yes |
| A9 Sessions | Close/reopen restores exactly where user left | v2 spec | yes |
| A10 No dark/light toggle | Single light theme this pass | scope | yes |
| A11 Dependencies | **USER OVERRIDE**: adopt the user's recommended stack — motion (framer), flexlayout or goldenlayout (dock), react-flow (graphs), zustand (state), tanstack query/table, cmdk, radix, shadcn/ui, tailwind, pixi/konva/d3/plotly as needed (REPLACES my prior zero-deps default) | explicit user stack message | yes |
| A12 src/ structure | Per user v3 folder layout: src/app/, shell/ (workspace/navigator/toolbar/inspector), panel-system/, window-manager/, layout/, plugins/ (attention/neurons/circuits/sae/logit-lens/embeddings/datasets/...), shared/ (stores/hooks/api/types/utils) | user v3 "I'd Also Change the Folder Structure" | yes |
| A13 Framework | **DECIDED: React** (user approved). Adopt the existing unmounted React tree (App.tsx, DockManager, LayoutManager, zustand-style store, design tokens, 700-line pages) as the live renderer; build Research Canvas on it; retire the Vue shell | user decision 2026-08-06 | irreversible (rewrite) |
| A14 Everything is a resource | Resource model (Change 3): Neuron, Attention Head, Circuit, Prompt, Dataset, Model, Paper, Experiment, SAE, Feature, Token — every resource opens inside a panel; backend returns data only, never "open panel/move window/select tab" (Change 10) | user v3 Change 3/10 | yes |
| A15 Local-only | Everything runs locally, not servers — no cloud; the existing local Python sidecar + SQLite remain the only backend | user v3 final line | no |

## Findings (cited - path:lines)
- **TWO frontends; the LIVE one is Vue.** index.html:8 loads `/src/main.ts` (Vue) -> App.vue (main.ts:1-8). React tree (App.tsx, main.tsx, 36 design/tokens imports) is never imported by the entry; main.tsx:1 is a self-reference. README.md describes a React app (stale doc).
- **The Vue shell is currently BROKEN token-wise.** 356+ .vue references to var(--ink/--primary/--hairline/--canvas/--ink-muted-*) — none defined in src/styles.css :root (only --bg-sidebar matches); the Apple-name tokens exist only in src/design/styles/global.css (imported by React files only, e.g. App.tsx:29).
- **styles.css: two conflicting :root blocks** (B&W minimal 5-30; "Light UI Overhaul" blue/gradient/glow 1735-1792) + a Tailwind-like utility shim ~1560-1690 mapping bg-slate-900 -> var(--bg) etc.
- **The React tree is the FEATURE-COMPLETE app** (pages 296-740 lines: NeuralExplorerView.jsx 738, Gpt2NeuronExplorer.tsx 701, TransformerVisualizer.tsx 740, ReasoningTraceView.jsx 562, Gpt2View.jsx 498) with DockManager, LayoutManager, panelRegistry, zustand-style store, design tokens. It is fully unmounted.
- **The Vue tree is a half-migrated shell**: only 4 real pages, ~25 stubs.
- **No floating/dock infra in either live path**; React's DockManager is dock-grid only (not floating); LayoutManager persists to localStorage.
- **Persistence backend exists**: electron/storage.js better-sqlite3 KV SettingsStore (get/set JSON) over IPC — fits layout/session persistence.
- **Tests**: Vitest targets React .jsx tests (vitest.config.js:8). Playwright shell-smoke asserts data-testid nav-<key> (Sidebar.vue:37) + header crumb labels for all 29 pages. If we switch the live renderer to React, the e2e tests must be re-pointed to the React shell (testids currently in Vue Sidebar) — a C9 risk to manage, not a blocker.

## Decisions (with rationale)
- D1 (RESOLVED): **React.** User approved. Flip index.html to /src/main.tsx, build the canvas on the existing React tree, retire the Vue shell after Playwright re-point. vue()/vueJsx() plugins may stay in vite.config (harmless) but the Vue source tree is retired from the entry.
- D2: Apple-light token layer aliasing both Apple names (--ink/--primary/--hairline/--canvas/--ink-muted-*) and legacy names (--bg/--text/--accent/--border) so all refs resolve.
- D3: Build Canvas/Panel/Window/Dock managers in React (FlexLayout/GoldenLayout per user stack as the dock engine, wrapped in our Canvas; ReactFlow for graphs).
- D4: Session/layout persistence via Electron better-sqlite KV + localStorage fallback.
- D5: Keep Playwright nav testids + breadcrumb labels green (compat layer or test re-point per framework choice).
- D6: Restructure src/ into the user's layered layout (A12) under the chosen framework.
- D7 (v3 manager split): NO god object. Distinct managers — WorkspaceManager, PanelManager, DockManager, LayoutManager, SelectionManager, CommandManager(Command Bus), WindowManager, PluginRegistry, SessionManager — each one responsibility, composed by the Application Shell.
- D8 (v3 resource flow): Workspace is permanent. Navigator tree -> double-click resource -> resource opens in workspace panel(s); backend returns analysis data only; UI decides presentation (Change 10).

## Scope IN
- Apple-light token layer (replace both :root blocks), typography/spacing/radius/shadow tokens.
- 5-region Research Canvas shell (Toolbar/Navigator/Workspace/Inspector/Bottom Workspace).
- Infinite workspace + Panel/Window/Dock managers: open/close/drag/resize/dock/split/pin/duplicate/detach/float, pan/zoom/snap, save+restore layouts.
- 14 spec panel types; mount real content where it exists (React pages carry into panels).
- Session persistence (windows/panels/camera/zoom/model/inspector/notes/timeline/console).
- Command palette (⌘K / Spotlight-style), notification center, animated interactions.
- Stub pages -> Navigator-reachable content/empty-state panels (no raw placeholders).
- User's layered src/ structure; adopt user's dependency stack (per A11, framework-dependent).
- Keep Vitest green; keep/manage Playwright shell-smoke.

## Scope OUT (Must NOT have)
- Do NOT touch backend Python API contract / Electron main behavior / IPC contract.
- Do NOT add dark/light toggle (single light theme).
- Stub pages: scaffolding/empty-state only, no fake features.
- Do NOT break Playwright nav testids + breadcrumbs (managed, not silently broken).
- NO "pages"/"tabs"/dashboard semantics — everything lives in the workspace (per spec, Change 1).
- NO server/cloud dependency — everything runs locally (Change final line; backend = local Python sidecar + SQLite only).
- Backend NEVER tells UI to open panel / move window / select tab (Change 10).
- React confirmed: the Vue tree is retired from the entry (deleted after Playwright re-point) — NOT mixed with React.

## Open questions
- None blocking. Framework (Q1) resolved: React (user approval 2026-08-06).

## Approval gate
status: awaiting-approval
Framework fork resolved (React). v2 spec + stack + v3 architecture changes (1-10, manager split, folder structure, local-only) all folded in. Next: explicit user approval of the brief -> write complete decision-complete plan (.omo/plans/redesign-ui-shell.md) + auto dual high-accuracy review (momus + independent oracle) before presenting. Execution stays separate (`$start-work`).