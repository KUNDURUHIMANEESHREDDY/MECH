# DesktopApp

An Electron + React + Python desktop application with a local SQLite store,
a JSON-lines IPC bridge to a Python sidecar, and a full test setup
(Vitest, pytest, Playwright) wired into GitHub Actions.

## Project layout

```
electron/        Electron main process, preload, IPC, storage, logger, Python bridge
src/             React renderer (Vite)
  components/    Sidebar, Topbar, pages for Workspace / Projects / Recent / etc.
  pages/         Settings sub-pages (Theme, GPU, Cache, Paths)
python/          Sidecar: main.py (JSON-lines server) + api.py (handlers)
storage/         SQLite database lives here at runtime
tests/
  pytest/        Python tests (unit + protocol end-to-end)
  vitest/        React / JS unit tests
  playwright/    E2E tests against the running Vite dev server
scripts/         dev.js, build.js, seed.js
.github/workflows/ci.yml
```

## Requirements

- Node.js 20+ and npm
- Python 3.9+ (the sidecar uses only the standard library; pytest is a dev dep)

## Install

```bash
npm install
pip install -r python/requirements.txt
```

## Develop

```bash
npm run dev
```

This starts the Vite dev server and launches Electron once Vite is
reachable. The renderer points at `http://localhost:5173`.

## Build

```bash
npm run build
```

Builds the renderer (`dist/`) and packages the app via `electron-builder`
into `release/`.

## Test

```bash
npm run test         # vitest + pytest
npm run test:js      # vitest only
npm run test:py      # pytest only
npm run test:e2e     # playwright
```

## How the pieces talk to each other

```
┌────────────────┐  contextBridge   ┌──────────────────┐
│  React (src)   │ ───────────────▶ │  Electron main   │
│  window.appApi │ ◀─────────────── │  (electron/*.js) │
└────────────────┘     IPC          └────────┬─────────┘
                                             │ JSON-lines over stdio
                                             ▼
                                    ┌──────────────────┐
                                    │  Python sidecar  │
                                    │  python/main.py  │
                                    └──────────────────┘
```

- **Settings** live in a SQLite DB at `storage/app.db` (`better-sqlite3`).
- **Projects** and **recent files** are table-backed in the same DB.
- The **Build** page spawns `npm run build:renderer` (or `python -m pip
  --version` for the Python target) and streams its output back to the
  renderer in real time.

## Definition of Done

- [x] **Electron launches** — `npm run dev` opens the window.
- [x] **Python connected** — `python/main.py` sidecar spawned on boot,
      pingable from the Workspace page.
- [x] **Settings saved** — Theme / GPU / Cache / Paths are persisted to
      SQLite via the `settings:get` / `settings:set` IPC handlers.
- [x] **Tests configured** — Vitest, pytest, and Playwright all wired
      up; CI runs them on every push.
