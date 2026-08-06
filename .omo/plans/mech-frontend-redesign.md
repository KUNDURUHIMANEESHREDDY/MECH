# mech-frontend-redesign - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** A complete, production-quality redesign of the frontend following Apple's design language, built with a reusable design system, organized components, and responsive layout.

**Why this approach:** Apple's design language provides a clean, professional aesthetic. Reusing shared tokens ensures consistency.

**What it will NOT do:** Will not change the backend API, will not remove existing functionality, will not break existing tests.

**Effort:** XL | Large
**Risk:** Medium - Large scope of changes across 30+ pages, risk of breaking existing tests.
**Decisions to sanity-check:** Light-only theme; VS Code-style IDE shell structure retained.

Your next move: approve the plan, then run `$start-work`. High-accuracy review running in parallel now.

Full execution detail follows below.

---

> TL;DR (machine): XL effort, Medium risk; redesign MECH frontend to Apple design language (light theme only) with reusable design token system over the 6-theme architecture, across all 30+ pages mapped as routes

## Scope
### Must have
- Create a new design system with Apple-style tokens (colors, typography, spacing, radii, elevation)
- Implement reusable UI primitives (Button, Input, Card, Typography, Layout)
- Restructure the app to follow Apple's aesthetic principles (SF Pro / Inter fonts, clean spacing)
- Apply the theme to all views/pages
- Ensure all existing tests pass

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No changes to backend API (localhost:8000)
- No removal of existing page functionality
- No breaking changes to existing test assertions (only additions/upgrades)
- No introduction of decorative gradients or shadows on chrome (per Apple design)
- No shadow on cards, buttons, or text (reserved for product imagery per Apple spec)

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after + framework (vitest run + playwright e2e + existing pytest)
- Evidence: .omo/evidence/task-<N>.md (screenshots, test outputs)

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.

- **Wave 1** (Foundation): design tokens, CSS variables, base reset, typography setup
- **Wave 2** (Primitives): Button, Input, Card, Typography, Layout primitives
- **Wave 3** (Shell): AppShell, ActivityBar, Sidebar, Topbar, ConsolePanel, StatusBar (reskinned)
- **Wave 4** (Pages, parallel): Core feature pages restyled (explorer, gpt2, gpt2explorer, transformer, transformerExplorer, neuralexplorer, workspace, models, prompts, debugger, benchmark, benchmarksuite, knowledgegraph, circuitexplorer, reasoning, evidencefusion, analytics, health, plugins, notebook, labnotebook, reproduction, campaigns, settings, logging, build, projects, recent, experiments, sessions, reports) — all 30+ pages mapped to routes
- **Wave 5** (Integration): App.tsx route wiring, store migration, final smoke tests

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| Tokens | — | Primitives, all rest | — |
| Primitives | Tokens | Shell, Pages | — |
| Shell | Primitives | Pages, App wiring | Pages |
| Pages | Primitives, Shell | App wiring | Shell |
| Integration | Pages, Shell | Final verification | — |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->
- [ ] 1. Create Apple design token system (colors.ts, typography.ts, spacing.ts, radii.ts, elevation.ts, index.ts)
  What to do / Must NOT do: Replace all inline colors/hex values with token references. Must NOT remove existing functionality.
  Parallelization: Wave 1 | Blocked by: — | Blocks: all remaining tasks
  References (executor has NO interview context - be exhaustive): frontend/src/styles.css:1-215 (existing themes), frontend/DESIGN.md:20-44 (existing tokens), Apple design doc (colors/typography/spacing/radii/elevation sections)
  Acceptance criteria (agent-executable): tsconfig compiles, tokens importable, no inline hex in new design files
  QA scenarios (name the exact tool + invocation): happy: run `npx tsc --noEmit -p tsconfig.json` in frontend/ and verify zero errors; failure: introduce a deliberate type error and confirm it fails, Evidence .omo/evidence/task-1.md
  Commit: Y | feat(frontend): add Apple design token system

- [ ] 2. Create CSS variable layer and global reset importing from tokens (design/styles/global.css)
  What to do / Must NOT do: Generate CSS custom properties from TypeScript tokens. Keep .app grid layout. Preserve VS Code IDE shell structure.
  Parallelization: Wave 1 | Blocked by: task 1 | Blocks: primitives, shell
  References (executor has NO interview context - be exhaustive): frontend/src/styles.css:215-250 (base/reset), frontend/src/styles.css:248-255 (.app grid), Apple design doc (colors: canvas, parchment, surface-tiles, ink, primary; typography: 17px body, SF Pro; elevation: flat/hairline/backdrop-blur/product-shadow)
  Acceptance criteria (agent-executable): global.css compiles via Vite build, :root has all token vars, scrollbars themed to Apple spec
  QA scenarios (name the exact tool + invocation): happy: `npm run build:renderer` succeeds with global.css included; failure: remove a required var and verify style build warning, Evidence .omo/evidence/task-2.md
  Commit: Y | feat(frontend): wire Apple design tokens to CSS variables

- [ ] 3. Implement UI primitives (Button.tsx, Input.tsx, Card.tsx, Typography.tsx, Layout.tsx) using design tokens
  What to do / Must NOT do: Implement button-primary (pill, #0066cc), button-secondary-pill, search-input, text-link as Apple-specified. Must NOT use decorative gradients or chrome shadows.
  Parallelization: Wave 2 | Blocked by: task 2 | Blocks: shell, pages
  References (executor has NO interview context - be exhaustive): frontend/src/components/ActivityBar.jsx:4-3 (lucide-react usage), frontend/DESIGN.md:46-117 (button/component specs), Apple design doc sections: Components (Buttons, Inputs, Cards), Shapes (radius scale), Elevation
  Acceptance criteria (agent-executable): Primitives render via Storybook-like test harness, button matches Apple #0066cc + pill radius
  QA scenarios (name the exact tool + invocation): happy: `npx vitest run tests/vitest/Primitives.test.jsx` (new test file) renders Button and asserts background #0066cc; failure: wrong color → assertion fails, Evidence .omo/evidence/task-3.md
  Commit: Y | feat(frontend): add Apple-style UI primitives

- [ ] 4. Reskin AppShell (ActivityBar, Sidebar, Topbar, ConsolePanel, StatusBar) using Apple tokens
  What to do / Must NOT do: Apply Apple tokens to existing IDE shell. Make ActivityBar collapsible. Make sidebar collapsible/ closable. Keep layout grid. Must NOT remove pages.
  Parallelization: Wave 3 | Blocked by: task 3 | Blocks: pages
  References (executor has NO interview context - be exhaustive): frontend/src/components/ActivityBar.jsx, frontend/src/components/Sidebar.jsx, frontend/src/components/Topbar.jsx, frontend/src/components/StatusBar.tsx, frontend/src/components/ConsolePanel.jsx, frontend/src/styles.css:248-400 (shell layout), frontend/src/App.tsx:359-398 (shell render)
  Acceptance criteria (agent-executable): Shell renders with Apple tokens, activity bar collapsible, sidebar collapsible
  QA scenarios (name the exact tool + invocation): happy: `npx playwright test tests/playwright/shell.e2e.ts` — toggle sidebar, assert width collapses; failure: toggle breaks → test fails, Evidence .omo/evidence/task-4.md
  Commit: Y | feat(frontend): reskin app shell with Apple tokens, collapsible panels

- [ ] 5. Restyle explorer page (Model Explorer) with Apple tokens
- [ ] 6. Restyle GPT-2 Live page with Apple tokens
- [ ] 7. Restyle GPT-2 Neuron Explorer page with Apple tokens
- [ ] 8. Restyle Transformer Visualizer page with Apple tokens
- [ ] 9. Restyle Transformer Explorer page with Apple tokens
- [ ] 10. Restyle Workspace/Campaign page with Apple tokens
- [ ] 11. Restyle Models page with Apple tokens
- [ ] 12. Restyle Prompts page with Apple tokens
- [ ] 13. Restyle Debugger page with Apple tokens
- [ ] 14. Restyle Experiments page with Apple tokens
- [ ] 15. Restyle Sessions page with Apple tokens
- [ ] 16. Restyle Reports page with Apple tokens
- [ ] 17. Restyle Neural Explorer page with Apple tokens
- [ ] 18. Restyle Benchmark Dashboard page with Apple tokens
- [ ] 19. Restyle Benchmark Suite page with Apple tokens
- [ ] 20. Restyle Knowledge Graph page with Apple tokens
- [ ] 21. Restyle Circuit Explorer page with Apple tokens
- [ ] 22. Restyle Reasoning Trace page with Apple tokens
- [ ] 23. Restyle Evidence Fusion page with Apple tokens
- [ ] 24. Restyle Campaigns page with Apple tokens
- [ ] 25. Restyle Research Analytics page with Apple tokens
- [ ] 26. Restyle Scientific Health page with Apple tokens
- [ ] 27. Restyle Plugin SDK page with Apple tokens
- [ ] 28. Restyle Research Notebook page with Apple tokens
- [ ] 29. Restyle Lab Notebook page with Apple tokens
- [ ] 30. Restyle Paper Reproduction page with Apple tokens
- [ ] 31. Restyle Projects page with Apple tokens
- [ ] 32. Restyle Recent Files page with Apple tokens
- [ ] 33. Restyle Settings page with Apple tokens
- [ ] 34. Restyle Logging page with Apple tokens
- [ ] 35. Restyle Build Log page with Apple tokens

Each page task (5-35):
  What to do / Must NOT do: Replace inline styles/classes with Apple token-based components. Preserve all data-binding to backend API. Must NOT alter API contract.
  Parallelization: Wave 4 (all parallelizable after task 4) | Blocked by: task 4 | Blocks: task 36
  References (executor has NO interview context - be exhaustive): frontend/src/App.tsx:59-91 (PAGES list), frontend/src/components/* (all .jsx/.tsx), frontend/src/panels/*, frontend/src/components/visualizations/*, frontend/src/components/visualizations/neuron-umap/*, existing vitest tests asserting page content
  Acceptance criteria (agent-executable): Each page renders with Apple tokens, no inline hex colors, all existing page tests green
  QA scenarios (name the exact tool + invocation): happy: `npx vitest run` — all 23 test files pass; `npx playwright test` — shell + smoke pass; failure: any test fails → investigate regression, Evidence .omo/evidence/task-<N>.md
  Commit: Y | feat(frontend): restyle all pages to Apple design language

- [ ] 36. Wire up App.tsx routing to new structure (React Router or path-based), migrate store if needed
  What to do / Must NOT do: Introduce a router so each page is a route. Migrate useAppStore if it helps. Must NOT break activePage navigation for tests.
  Parallelization: Wave 5 | Blocked by: tasks 1-35 | Blocks: final verification
  References (executor has NO interview context): frontend/src/App.tsx (full), frontend/src/store/useAppStore.ts, frontend/src/hooks/useModel.ts, frontend/src/services/api.ts, tests/vitest/Sprint*.test.jsx
  Acceptance criteria (agent-executable): All PAGES map to routes, navigation works, all 23 vitest files pass
  QA scenarios (name the exact tool + invocation): happy: `npm run test:js` passes all 23; `npx playwright test --grep "navigation"` passes; failure: route mismatch → test fails, Evidence .omo/evidence/task-36.md
  Commit: Y | refactor(frontend): wire routes and store to Apple-redesigned shell

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit — Verify all 6 Apple themes implemented or only light (scope: light only). Confirm tokens used everywhere, no inline hex remaining. Check .omo/evidence/task-final-audit.md
- [ ] F2. Code quality review — tsc --noEmit zero errors, eslint passes, no `any` introduced
- [ ] F3. Real manual QA — Playwright headless: smoke all 30 routes render, shell collapsible toggles work, command palette works
- [ ] F4. Scope fidelity — All 30 pages retained, API contract unchanged, test count >= 23 passing

## Commit strategy
- One commit per task (1-36), atomic and self-contained
- Commit messages follow: `feat(frontend): ...`, `refactor(frontend): ...`
- No squashing of intermediate work; each task independently verifiable
- Only commit when explicitly green (tests pass)

## Success criteria
- `npm run test:js` — 23/23 vitest files pass
- `npx playwright test` — all e2e pass
- `npx tsc --noEmit` — zero type errors
- `npm run build:renderer` — builds successfully
- All 30 pages map to routes and render
- Activity bar and sidebar collapsible
- Apple design tokens applied consistently (no inline hex in component code)
- Zero breaking changes to backend API or existing test assertions
