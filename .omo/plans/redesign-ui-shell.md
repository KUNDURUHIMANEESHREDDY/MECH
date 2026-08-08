---
slug: redesign-ui-shell
status: approved
review_required: true
plan_path: .omo/plans/redesign-ui-shell.md
intent: redesign
goal: Rebuild the MECH frontend into an Apple-light Visual Research Canvas desktop app on the React renderer, with a permanent resource-based workspace, hierarchical Navigator, panel registry + plugin contract, command bus, session persistence, and an App Shell built from nine single-responsibility managers. No pages/tabs/dashboard semantics. Everything runs locally.
verification: npm run test:js green (React vitest already targets .jsx/.tsx), Playwright e2e green after re-point, npm run build:renderer exit 0, manual shell smoke in the packaged renderer. LSP diagnostics clean on every edited file.
scope-in: token layer, 5-region canvas shell, multi-workspace, panel/dock/window/layout/selection/command/plugin/session managers, command palette, panel plugin registrations mapping to existing React content, Playwright re-point, deletion of retired Vue tree.
scope-out: backend Python API contract, Electron main behavior, IPC contracts, dark mode, fake features, cloud/server.
---

# Goal

Rebuild the MECH React frontend into an Apple-light **Visual Research Canvas** desktop app. The shell is a **permanent workspace** — no pages, no tabs, no dashboard. Users open **resources** (models, neurons, attention heads, circuits, prompts, datasets, papers, experiments, SAEs, features, tokens, sessions) from a **hierarchical Finder-like Navigator**; each resource opens inside a dockable panel on the infinite workspace. Everything runs locally.

# Context (verified)

- **Live renderer today is a Vue 3 shell**: `frontend/index.html:10` loads `/src/main.ts` → `App.vue`. Vue chrome is duplicated as `.vue` files (`Sidebar.vue`, `Topbar.vue`, `ActivityBar.vue`, `StatusBar.vue`, ...).
- **The React tree is feature-complete and self-contained**: `App.tsx` (439 lines) imports `Sidebar`/`Topbar`/`ActivityBar`/`StatusBar`/`ConsolePanel`/`DockManager`/`CommandPalette` + all pages + the Apple design system (`src/design/`), and resolves to the **`.jsx` variants** (e.g. `Sidebar.jsx`). `src/main.tsx` exists as the React entry (`ReactDOM.createRoot → <App/>`); `App.tsx` imports `./design/styles/global.css` and `./styles.css`.
- `vite.config.mts:9` already wires `vue()`, `react()`, `vueJsx()`, `tailwindcss()` — the React path needs **no new build wiring**; only `index.html` switches to `/src/main.tsx`.
- **Two token naming schemes collide**: Vue components use bare `var(--ink)`, `var(--primary)`, `var(--hairline)`, `var(--canvas)`, `var(--ink-muted-48/80)` (~356 refs); the React design system uses prefixed `--color-ink`, `--color-primary`, `--color-hairline`, `--color-canvas` (from `src/design/styles/global.css`). The single token layer must define **both** spellings. `styles.css` has two conflicting `:root` blocks (B&W at lines 5–30; "Light UI Overhaul" blue/glow/gradient at 1735–1792) + a utility shim (~1560–1690).
- **Persistence exists**: `electron/storage.js` provides `SettingsStore` (KV get/set JSON), `SessionStore` (sessions table + metadata JSON), better-sqlite3 at `storage/app.db`, WAL. Exposed over IPC via preload/contextBridge (`window.appApi`). The renderer needs only a thin adapter — nothing in Electron main changes.
- **Existing React panel content**: `src/panels/*.tsx` (TokenPanel, LayerPanel, PredictionPanel), `src/components/visualizations/panels/*.tsx` (AttentionHeatmap, ActivationHeatmap, TokenViewer), `src/components/visualizations/neuron-umap/*` (NeuronUMAP, useNeuronUMAP), plus ~30 `.jsx` pages (`Gpt2View`, `Gpt2NeuronExplorer`, `TransformerVisualizer`, `NeuralExplorerView`, `CircuitExplorerView`, `KnowledgeGraphView`, ...). These become panel bodies behind the registry.
- **Existing infra to build on**: `src/layout/DockManager.tsx` (grid + toggle dock, NOT floating), `src/layout/LayoutManager.ts` (localStorage only), `src/services/panelRegistry.ts` (panel definitions with id/title/icon/defaultDock/commands). Germ of the new system — extended, not thrown away.
- **Existing state**: `src/store/useAppStore.ts` — zustand-style external store with `AppState` (activePage, workspace, selection, visiblePanels, commandPaletteOpen). Keep, evolve to workspace-as-state.
- **Tests**: `vitest.config.js:16` includes `tests/vitest/**/*.test.{js,jsx}` (React-only). `tests/playwright/shell-smoke.spec.js` asserts `data-testid="nav-<key>"` for 29 nav keys + header crumb labels + activity bar toggle + `python-status` testid — must stay green against the **React** shell.

# Decisions (binding)

- D1 **React is the renderer**. Flip `index.html` to `/src/main.tsx`. Retire the Vue tree (delete `App.vue`, `main.ts`, all `.vue` files, `pinia` usage) AFTER the Playwright re-point is green. Never mix trees. `vue()`/`vueJsx()` plugins and `@vitejs/plugin-vue`/`@vitejs/plugin-vue-jsx`/`vue`/`pinia`/`lucide-vue-next` deps may be removed once no `.vue` file remains.
- D2 **Apple-light single token system** in `src/styles.css`. ONE `:root` block replaces BOTH existing blocks. Define Apple tokens with **both** spellings (`--color-ink` + `--ink`, `--color-primary` + `--primary`, `--color-hairline` + `--hairline`, `--color-canvas` + `--canvas`, `--ink-muted-48/80`, `--ink-muted-48`). No gradients, no glass, no neon, no glow. Light theme only (no toggle). Body bg `#f5f5f7`, canvas white. Add `--bg`/`--text`/`--accent`/`--border` legacy aliases.
- D3 **Application Shell** (5 regions): Toolbar, Navigator, Workspace, Inspector, Bottom Workspace. Composed by `App.tsx` (kept in `src/app/`). Shell holds: `Toolbar` (global: logo/search/model/session/Run/Export/Workspace/Settings/Profile), `Navigator` (hierarchical tree), `Workspace` (infinite canvas), `Inspector` (context-sensitive right), `BottomWorkspace` (tabbed Console/Timeline/Logs/Notes/Results/Terminal/Exports/Errors/Tasks).
- D4 **Nine single-responsibility managers, no god object**:
  1. `WorkspaceManager` — owns the Workspace region, canvas camera/zoom, open windows
  2. `PanelManager` — owns panel instances (from registry), open/close/focus
  3. `DockManager` — dock zones, split, snap (FlexLayout engine)
  4. `LayoutManager` — serializable layout (positions, sizes, dock assignments)
  5. `SelectionManager` — shared selection context (resource, layer, head, neuron, token)
  6. `CommandManager` — command bus (publish/subscribe + command palette source)
  7. `WindowManager` — floating/detached window model (position, size, z-order)
  8. `PluginRegistry` — panel plugin registration (name→PanelPlugin)
  9. `SessionManager` — workspace-as-state snapshot/restore, persistence
- D5 **Session persistence** via electron `appApi` KV (`settings:get`/`settings:set` or `sessions` columns) through `src/shared/api/persistence.ts`, with localStorage fallback for non-Electron dev. Save ONE workspace object → restore windows/panels/camera/model/inspector/notes/timeline/console.
- D6 **Playwright re-point**: `tests/playwright/shell-smoke.spec.js` nav `data-testid="nav-<key>"` + header crumb labels must stay green against the React shell. The React `Topbar.jsx`/`Sidebar.jsx` must render crumb labels matching `PAGES` keys and `nav-<key>` testids. Where a testid exists only in the Vue chrome, add it to the React component. Keep vitest green (already React).
- D7 **New folder structure** (user v3): see structure in Components/Phases; migrate the React tree into it atomically (one commit), keeping imports working via path updates.
- D8 **Resource flow**: Navigator tree → double-click resource (or ⌘K) → resource opens a panel in the Workspace. Workspace is permanent; nothing navigates. Backend returns data only; UI decides presentation (Change 10).
- D9 **Tool palette**: ⌘K Spotlight (cmdk) — search resources/panels/commands, Enter opens.
- D10 **Multi-workspace**: named workspaces (GPT-2 / Gemma / Llama), instant switch, per-workspace layout + sessions (Change 9).

# Components (outcomes ledger)

| id | outcome | verification |
|----|---------|--------------|
| C1 token-system | Single Apple-light `:root` in styles.css; both spellings; legacy aliases; both old :root blocks gone | grep `--color-ink` and `--ink` both resolve; no `linear-gradient`/`backdrop-filter`/`box-shadow: 0 0` in :root; e2e green |
| C2 shell | 5-region Research Canvas shell in React (App.tsx + Shell) | shell renders in dev; navigator/toolbar/inspector/bottom visible; e2e nav+crumb green |
| C3 workspace | Infinite canvas + PanelManager + WindowManager + DockManager: open/close/drag/resize/dock/split/pin/detach/float, pan/zoom/snap, save+restore | drag a panel to a dock zone snaps; float a panel to a window; reload restores layout |
| C4 inspector | Context-sensitive right Inspector via SelectionManager + CommandBus | select neuron → inspector updates |
| C5 bottom | Tabbed bottom: Console/Timeline/Logs/Notes/Results/Terminal/Exports/Errors/Tasks | tabs render; console receives command-bus events |
| C6 sessions | Workspace-as-state snapshot/restore (windows/panels/camera/model/inspector/notes/timeline/console) | close app, reopen → identical layout; saved via KV + localStorage fallback |
| C6b multi-workspace | Named workspaces with instant switch, per-workspace layouts | switch A→B→A restores each |
| C7 panels-content | Registered panel plugins mount REAL content (AttentionHeatmap, NeuronUMAP, Gpt2View, ...) | each panel type renders real data when model loaded |
| C8 stub-pages | Legacy flat pages → navigator resource entries / empty-state panels; no fake features | 29 nav keys present in navigator or palette; no broken links |
| C8b registry | PanelRegistry accepts plugin registration; new analysis = registration only | register a dummy plugin → appears in palette + navigator |
| C8c panel-contract | PanelPlugin interface: Header/Toolbar/Body/Inspector/Context Menu/Commands/Persistence | every plugin implements the contract typecheck-clean |
| C8d command-bus | CommandManager publish/subscribe; NeuronSelected → Inspector+Timeline+Circuit+Console+Notes | click neuron → 4 subscribers update |
| C9 tests | Vitest green; Playwright shell-smoke green against React shell | npm run test:js && npx playwright test |
| C10 react-tree | index.html → /src/main.tsx; Vue tree deleted; vue deps removed | npm run build:renderer exit 0; no .vue files remain |
| C11 tool-palette | ⌘K Spotlight (cmdk): search resources/panels/commands, Enter opens | type "attention" → Enter → panel opens |
| C12 folders | React tree migrated to v3 structure | npm run test:js green post-move |

# Target structure (D7)

```
src/
  app/                      # entry composition, <App/> root, Shell
  shell/
    workspace/              # Workspace region (infinite canvas)
    navigator/              # hierarchical resource tree
    toolbar/                # global Toolbar region
    inspector/              # context Inspector region
    bottom/                 # tabbed bottom workspace
  panel-system/             # PanelPlugin contract, PanelRegistry
  window-manager/           # WindowManager, floating window model
  layout/                   # DockManager, LayoutManager (FlexLayout)
  plugins/
    attention/  neurons/  circuits/  sae/  logit-lens/  embeddings/  datasets/
  shared/
    managers/               # Workspace/Panel/Selection/Command/Session managers
    stores/                 # zustand stores (workspace, selection, ui)
    hooks/
    api/                    # persistence adapter, python bridge
    types/                  # Resource, PanelPlugin, WorkspaceState, ...
    utils/
```

# Execution phases (ordered; each ends with its verification gate)

## Phase 0 — Baseline (gate: current tests green)
1. Run `npm run test:js` and `npm run test:e2e` in `frontend/`. Record baseline (expected: vitest green, e2e green against Vue shell today).
2. Commit baseline note to the work branch (branch: `redesign/visual-research-canvas`).

## Phase 0.5 — Install stack dependencies (gate: npm install clean, build still green)
VERIFIED: `package.json` has NO zustand/cmdk/flexlayout-react/@tanstack-*/motion — only `react`, `react-dom`, `reactflow`, `lucide-react` (React side). `zustand` exists in node_modules only as a transitive artifact; the current React store (`src/store/useAppStore.ts`) is a hand-rolled external store, not zustand.
1. Install per user-approved stack (A11/D5): `npm install zustand cmdk flexlayout-react @tanstack/react-query @tanstack/react-table motion @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-tabs @radix-ui/react-tooltip class-variance-authority clsx tailwind-merge lucide-react` (lucide-react already present; Radix/TVA/clsx/tw-merge needed for shadcn-style components). React 18.3.1 stays (no React 19 upgrade).
2. Re-run `npm run build:renderer` + `npm run test:js` — gate: exit 0, no new failures.
3. Do NOT add Tailwind-theming shadcn unless needed for a specific shell component; the Apple token layer (Phase 2) is the styling source of truth.

## Phase 1 — Flip renderer to React (C10) (gate: React shell boots, e2e green against React)
1. Edit `frontend/index.html:10`: `src="/src/main.ts"` → `src="/src/main.tsx"`.
2. Start `npm run dev:renderer`. Verify the **React** app boots at :5173 (App.tsx chrome: ActivityBar/Sidebar/Topbar/StatusBar/ConsolePanel visible).
3. Run e2e: `npx playwright test`. Fix any testid/crumb gaps in the React `Topbar.jsx`/`Sidebar.jsx` so all 29 `nav-<key>` + crumb labels + `python-status` + activity-bar tests pass **against the React shell**.
4. Gate: `npx playwright test` green with React renderer live.

## Phase 2 — Token system (C1) (gate: grep check + e2e still green)
1. Replace BOTH `:root` blocks in `src/styles.css` (lines 5–30 and 1735–1792) with ONE Apple-light block.
2. Define both spellings: `--color-*` (React) and bare `--*` (Vue) for ink/primary/hairline/canvas/ink-muted-48/ink-muted-80; legacy aliases `--bg`/`--text`/`--accent`/`--border`. Keep the utility shim (~1560–1690) but retarget its colors to the new tokens.
3. Delete or neutralize `src/design/styles/global.css`'s duplicate `:root` if it conflicts (keep the design-token TS exports as the source of truth; alias CSS vars from them).
4. Gate: no undefined var refs (grep the 356+ bare names — all defined); e2e green.

## Phase 3 — Core stores + command bus (C8d) (gate: typecheck + vitest green)
1. `src/shared/stores/selection.ts` — zustand: selected resource/layer/head/neuron/token. Backed by `SelectionManager`. (zustand installed in Phase 0.5; migrate the hand-rolled `src/store/useAppStore.ts` store to zustand so there is ONE state lib).
2. `src/shared/managers/commandManager.ts` — typed event bus: `publish(event, payload)`, `subscribe(type, handler)`. Channels: `resource.opened`, `neuron.selected`, `token.hovered`, `circuit.highlighted`, `console.log`, `note.created`, `timeline.event`, `layout.changed`.
3. Wire SelectionManager → publish on change.
4. Gate: unit test publishes `neuron.selected` and asserts 3 subscribers fire.

## Phase 4 — Panel system + registry (C8b, C8c) (gate: registry test green)
1. `src/shared/types.ts` — `Resource` (kind/id/label/metadata), `PanelPlugin` contract:
   `{ id, title, icon, category, defaultDock, resourceKinds: ResourceKind[], Header?, Toolbar?, Body, Inspector?, contextMenuItems, commands[], persist(workspace) }`.
2. `src/panel-system/pluginRegistry.ts` — register/get/list by id, kind, category.
3. Port the 7 existing `panelRegistry` definitions into plugins (token_viewer, attention_heatmap, activation_heatmap, neuron_panel, layer_inspector, prediction_inspector, token_inspector) mapping to the real React bodies.
4. Gate: registry unit test; a dummy plugin registers and lists.

## Phase 5 — Shell + workspace (C2, C3) (gate: shell boots, drag/dock/float works)
1. `src/app/Shell.tsx` — 5-region layout: Toolbar / Navigator / Workspace / Inspector / BottomWorkspace.
2. `src/shell/workspace/` — infinite canvas: pan/zoom (wheel+space-drag), snap-to-grid, camera state in WorkspaceManager.
3. `src/layout/DockManager.tsx` (extend existing) — dock zones (left/right/bottom/center), split, tab-stack, drag-to-dock with drop preview. Use FlexLayout (`flexlayout-react`) as the engine per user stack.
4. `src/window-manager/WindowManager.tsx` — floating windows: drag title bar, resize, z-order, minimize/close; detach a docked panel → float; re-dock a floating window.
5. Gate: manual — drag a panel to right zone snaps; detach floats; re-dock; pan/zoom the canvas; e2e still green.

## Phase 6 — Navigator + resource flow (D8, C8) (gate: double-click opens panel)
1. `src/shell/navigator/` — hierarchical tree: Workspace/Research/Sessions, Models (GPT-2/Gemma/Llama), Assets (Datasets/Exports/Layouts), Bookmarks; Finder-style (chevrons, selection, double-click).
2. Resource list source: local registry of known resources + backend capability query (data only, Change 10).
3. Double-click resource → CommandManager publishes `resource.opened` → PanelManager opens matching plugin panel (resourceKinds match) + Console logs + Timeline event + Inspector updates.
4. Legacy flat pages → navigator entries under "Workspace" group; stub pages → empty-state panels (no fake features).
5. Gate: dbl-click "GPT-2 Small" opens Attention Map + Neuron Inspector + Console; Inspector shows selection.

## Phase 7 — Inspector + bottom workspace (C4, C5) (gate: selection-driven updates)
1. `src/shell/inspector/` — reads SelectionManager; renders context form based on selected resource (model/layer/neuron/token/circuit).
2. `src/shell/bottom/` — tabs: Console, Timeline, Logs, Notes, Results, Terminal, Exports, Errors, Tasks. Console subscribes to `console.log` bus; Timeline to `timeline.event`; Notes editable + persisted.
3. Gate: click neuron → Inspector updates, Timeline records, Console logs, Notes mention (per Change 6 example).

## Phase 8 — Session persistence + multi-workspace (C6, C6b, D5, D10) (gate: restore round-trip)
1. `src/shared/api/persistence.ts` — electron `appApi.settings.get/set` (KV JSON); fallback localStorage when `window.appApi` absent.
2. `SessionManager` — snapshot: `{ workspaceId, layout, windows, camera, model, selection, notes, timeline, console }`; save on change (debounced), restore on boot; per-workspace snapshots.
3. Workspaces A/B/C switcher in Toolbar (Workspace menu); instant switch restores layout+session each.
4. Gate: open panels, move/float, zoom, write note → reload (or `appApi.set`/`get` round-trip in dev) → everything returns exactly.

## Phase 9 — Tool palette (C11, D9) (gate: ⌘K works)
1. `src/shell/toolbar/CommandPalette.tsx` (extend existing) — cmdk-based Spotlight: indexes resources, panel plugins, and CommandManager commands.
2. ⌘K toggles; type "attention" → Enter → `resource.opened` → panel opens; keyboard nav.
3. Gate: ⌘K → "attention" → Enter opens Attention Map.

## Phase 10 — Playwright parity + full test pass (C9) (gate: all tests green)
1. Sweep `shell-smoke.spec.js` against final shell — fix testids/crumbs in React components; keep all 29 nav keys + labels + `python-status` + activity toggle assertions passing.
2. Run `npm run test:js` (React vitest) — fix regressions.
3. Run `npm run build:renderer` — exit 0.
4. Gate: `npm run test:js && npx playwright test && npm run build:renderer` all green.

## Phase 11 — Retire Vue tree (C10 finish) (gate: no .vue files; build green)
1. Delete: `src/App.vue`, `src/main.ts`, all `src/components/**/*.vue`, `src/design/**/*.vue` if any, vue-only files (`vite-env` vue refs). Keep `.tsx`/`.jsx`/`.ts` React equivalents.
2. Remove Vue plugins/deps: `vue()`, `vueJsx()` from `vite.config.mts`; remove `@vitejs/plugin-vue`, `@vitejs/plugin-vue-jsx`, `vue`, `pinia`, `lucide-vue-next` from `package.json`; `npm install` to prune.
3. Full test pass again (vitest + playwright + build).
4. Gate: `Get-ChildItem -Recurse -Filter *.vue` returns nothing; all suites green.

## Phase 12 — Folder restructure to v3 layout (C12) (gate: tests green post-move)
1. Move React source into `src/app`, `src/shell/*`, `src/panel-system`, `src/window-manager`, `src/plugins/*`, `src/shared/*` per D7. Update imports (`@/` alias exists).
2. Commit the restructure separately AFTER shell is green (one atomic move commit).
3. Full test pass.
4. Gate: `npm run test:js && npx playwright test && npm run build:renderer` green; structure matches D7.

# Definition of Done (all must hold)
- [x] `npm run test:js` green
- [x] `npx playwright test` green against the React shell (29 nav keys + crumbs + python-status + activity toggle)
- [x] `npm run build:renderer` exit 0; `npm run dev` boots the React app
- [x] No `.vue` files remain; Vue deps removed
- [x] Apple-light single token layer (no gradients/glass/neon); both token spellings resolve
- [x] 5-region shell: Toolbar/Navigator/Workspace/Inspector/Bottom Workspace
- [x] Resource flow: dbl-click or ⌘K opens panel; nothing navigates away
- [x] Panels: drag/dock/split/float/pin/close; layout save+restore
- [x] Command bus: neuron.selected updates Inspector+Timeline+Console+Notes
- [x] Multi-workspace A/B/C instant switch, per-workspace session restore
- [x] Backend contract untouched; Electron main untouched; IPC untouched; everything local

# Risks & mitigations
- **Token double-scheme drift** → single source in `design/tokens/*.ts`; CSS vars generated/aliased from it; grep gate in Phase 2.
- **FlexLayout vs existing DockManager churn** → wrap FlexLayout behind our DockManager interface; existing `dock-panel` CSS reused.
- **e2e flakiness on renderer switch** → Phase 1 fixes testids against React BEFORE any feature work; retry loop `toPass` already in spec.
- **Folder move breaking imports** → `@/` alias; atomic commit; full test pass immediately after.
- **Persistence unavailable (no Electron)** → localStorage fallback keeps dev working (D5).

# Approval / execution
- Plan status: approved (user "go", 2026-08-06). Execution is a separate step via `$start-work` — this file is the single source of truth for the executor.

