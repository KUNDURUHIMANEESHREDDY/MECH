'use strict';

const electron = require('electron');

// `ELECTRON_RUN_AS_NODE=1` makes electron.exe run as a plain Node process.
// `require('electron')` then resolves to the npm package -- a path string --
// so every API below is undefined and the next line dies with a TypeError
// that says nothing about the cause. Name the cause instead.
if (!electron || !electron.app) {
  console.error('[mech] ELECTRON_RUN_AS_NODE is set, so Electron ran as plain Node.');
  console.error('[mech] Unset it (set ELECTRON_RUN_AS_NODE=) and start the app again.');
  process.exit(1);
}

const { app, BrowserWindow, ipcMain, shell, dialog, session } = electron;
const path = require('node:path');
const { spawn } = require('node:child_process');
const http = require('node:http');
const fs = require('node:fs');
const { registerIpcHandlers } = require('./ipc');
const { resolveApiToken } = require('./apiProxy');
const crypto = require('node:crypto');
const { getLogger } = require('./logger');
const { getStorage } = require('./storage');

const isDev = !app.isPackaged;
const RENDERER_DEV_URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';
const BACKEND_PORT = 8000;
const BACKEND_HOST = '127.0.0.1';
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const BACKEND_STARTUP_TIMEOUT_MS = 180000;

let mainWindow = null;
let pythonBackend = null;

/**
 * A backend that cannot start is a startup failure, not a degraded mode.
 *
 * Carrying a message out of `startBackend` lets bootstrap put it in front of
 * the user (dialog + log) instead of leaving a window that silently 401s.
 */
class BackendStartupError extends Error {}

function createMainWindow() {
  const logger = getLogger();

  const win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 600,
    show: false,
    backgroundColor: '#0e1116',
    title: 'MECH Platform',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      // The renderer is untrusted code. The bearer token lives in the main
      // process and is reachable only through the `api:request` proxy, so the
      // renderer has no reason to need Node or a privileged preload.
      sandbox: true,
      spellcheck: false
    }
  });

  win.once('ready-to-show', () => {
    win.show();
    logger.info('main_window_ready', { width: win.getBounds().width });
  });

  win.webContents.on('console-message', (_event, level, message) => {
    logger.info('renderer_console', { level, message });
  });
  win.webContents.on('did-fail-load', (_event, code, desc) => {
    logger.error('renderer_load_failed', { code, desc });
  });

  // External links open in the OS browser; the app never navigates away.
  win.webContents.setWindowOpenHandler(({ url }) => {
    try {
      const target = new URL(url);
      if (target.protocol === 'https:' || target.protocol === 'http:') {
        shell.openExternal(url).catch(() => undefined);
      }
    } catch {
      /* not a URL we will hand to the OS */
    }
    return { action: 'deny' };
  });

  // No <webview> embeds: they would be a second, unvalidated renderer.
  win.webContents.on('will-attach-webview', (event) => {
    logger.warn('webview_attach_denied', {});
    event.preventDefault();
  });

  if (isDev) {
    win.loadURL(RENDERER_DEV_URL).catch((err) => {
      logger.error('renderer_load_failed', { error: err.message });
    });
  } else {
    const indexPath = path.join(__dirname, '..', 'dist', 'index.html');
    win.loadFile(indexPath).catch((err) => {
      logger.error('renderer_load_failed', { path: indexPath, error: err.message });
    });
  }

  win.on('closed', () => {
    if (mainWindow === win) mainWindow = null;
  });

  return win;
}

/**
 * Locate the interpreter that runs the backend.
 *
 * `bundledOnly` is the packaged-app rule: the only acceptable interpreters are
 * the one this build shipped (`resources/python/python.exe`) or an explicit
 * operator override (`MECH_PYTHON`). Falling back to bare `python` in a
 * release means the app quietly depends on whatever the machine happens to
 * have -- the exact non-portability this packaging exists to remove.
 */
function resolvePythonPath(resourcesPath, { bundledOnly = false } = {}) {
  const candidates = [];

  if (process.env.MECH_PYTHON) candidates.push(process.env.MECH_PYTHON);

  // Bundled runtime, produced by scripts/python_runtime.js and copied by
  // electron-builder's `extraResources` entry ({ from: "build/python" }).
  candidates.push(path.join(resourcesPath, 'python', 'python.exe'));

  if (!bundledOnly) {
    candidates.push(path.join(resourcesPath, '.venv', 'Scripts', 'python.exe'));
    if (process.env.PYTHON_PATH) candidates.push(process.env.PYTHON_PATH);
    // Development / local checkout: use the project's virtual environment.
    const repoRoot = path.join(__dirname, '..', '..');
    candidates.push(path.join(repoRoot, '.venv', 'Scripts', 'python.exe'));
    candidates.push(path.join(repoRoot, 'venv', 'Scripts', 'python.exe'));
  }

  for (const candidate of candidates) {
    try {
      if (candidate && fs.existsSync(candidate)) return candidate;
    } catch { }
  }

  if (bundledOnly) return null;
  return process.platform === 'win32' ? 'python' : 'python3';
}

function probeHealth() {
  return new Promise((resolve, reject) => {
    const req = http.get(`${BACKEND_URL}/health`, (res) => { res.resume(); resolve(res); });
    req.on('error', reject);
    req.setTimeout(2000, () => { req.destroy(new Error('timeout')); });
  });
}

async function startBackend(logger) {
  try {
    const res = await probeHealth();
    if (res.statusCode && res.statusCode < 500) {
      logger.info('backend_reuse', { url: BACKEND_URL });
      if (!resolveApiToken()) {
        // The backend on this port was not started by this build and its
        // token is not readable here, so every proxied request will fail
        // with "API token unavailable". Say so once, at startup.
        logger.error('backend_token_unresolved', {
          hint: 'set MECH_API_TOKEN to the token the reused backend accepts'
        });
      }
      return null;
    }
  } catch {
  }

  let pythonPath, scriptPath, cwd, pythonPathEnv;

  if (isDev) {
    const repoRoot = path.join(__dirname, '..', '..');
    pythonPath = resolvePythonPath(repoRoot);
    scriptPath = path.join(repoRoot, 'backend', 'main.py');
    cwd = repoRoot;
    pythonPathEnv = `${repoRoot}${path.delimiter}${path.join(repoRoot, 'backend')}`;
  } else {
    const resourcesPath = process.resourcesPath || path.join(__dirname, '..');
    pythonPath = resolvePythonPath(resourcesPath, { bundledOnly: true });
    if (!pythonPath) {
      throw new BackendStartupError(
        'No bundled Python runtime was found in this installation.\n\n' +
        `Expected: ${path.join(resourcesPath, 'python', 'python.exe')}\n\n` +
        'This build was packaged without its Python runtime, so the research ' +
        'backend cannot start. Reinstall MECH, or set MECH_PYTHON to an ' +
        'interpreter that has the backend requirements installed.'
      );
    }
    scriptPath = path.join(resourcesPath, 'backend', 'main.py');
    cwd = resourcesPath;
    pythonPathEnv = `${resourcesPath}${path.delimiter}${path.join(resourcesPath, 'backend')}`;
  }

  if (!fs.existsSync(scriptPath)) {
    throw new BackendStartupError(
      `The backend entry point is missing from this installation: ${scriptPath}`
    );
  }

  // The backend and this process must agree on one token. The backend prefers
  // MECH_API_TOKEN over the file it writes, and the renderer proxy prefers
  // the same variable, so sharing it here means a packaged install
  // authenticates even where the token file cannot be written.
  const tokenFromFile = resolveApiToken();
  if (tokenFromFile) {
    process.env.MECH_API_TOKEN = tokenFromFile;
  } else if (!isDev && !process.env.MECH_API_TOKEN) {
    // No token file: a fresh packaged install, or one whose storage
    // directory is not writable. The token travels in the child's
    // environment instead. In development the backend writes the file
    // itself and a browser session reads that same file.
    process.env.MECH_API_TOKEN = crypto.randomBytes(32).toString('base64url');
  }
  logger.info('backend_spawning', { pythonPath, scriptPath, cwd, pythonPathEnv });

  const stderrTail = [];
  const child = spawn(pythonPath, [scriptPath], {
    cwd,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONPATH: pythonPathEnv, PYTHONUNBUFFERED: '1' },
    windowsHide: true,
  });

  child.stdout?.on('data', (d) => logger.info('backend_stdout', { line: d.toString().trim() }));
  child.stderr?.on('data', (d) => {
    const line = d.toString().trim();
    logger.warn('backend_stderr', { line });
    stderrTail.push(line);
    if (stderrTail.length > 40) stderrTail.shift();
  });
  child.on('exit', (code) => logger.info('backend_exited', { code }));

  let spawnError = null;
  child.on('error', (err) => {
    spawnError = err;
    logger.error('backend_spawn_error', { error: err.message });
  });

  // Packaged local Python environments can take longer on first launch while
  // Windows scans/imports native ML packages. Wait so the UI opens connected.
  const deadline = Date.now() + BACKEND_STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (spawnError) {
      throw new BackendStartupError(
        `The backend process could not be started (${pythonPath}).\n\n${spawnError.message}`
      );
    }
    if (child.exitCode !== null) {
      throw new BackendStartupError(
        `The backend exited during startup with code ${child.exitCode} (${pythonPath}).\n\n` +
        `${stderrTail.slice(-12).join('\n') || 'No error output was produced.'}`
      );
    }
    try {
      const res = await probeHealth();
      if (res.statusCode && res.statusCode < 500) {
        logger.info('backend_ready', { url: BACKEND_URL });
        return child;
      }
    } catch { }
    await new Promise((r) => setTimeout(r, 500));
  }

  throw new BackendStartupError(
    `The backend did not become ready within ${Math.round(BACKEND_STARTUP_TIMEOUT_MS / 1000)}s (${pythonPath}).\n\n` +
    `${stderrTail.slice(-12).join('\n') || 'No error output was produced.'}`
  );
}

async function stopBackend(child, logger) {
  if (!child || child.killed) return;
  try {
    child.kill();
  } catch (e) {
    logger.error('backend_kill_error', { error: e.message });
  }
}

async function bootstrap() {
  const logger = getLogger();
  logger.info('app_boot', { version: app.getVersion(), isDev });

  // No permission is granted to any page by default; the app needs none.
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => {
    callback(false);
  });

  const storage = getStorage();
  await storage.init();
  logger.info('storage_ready', { db: storage.dbPath });

  try {
    pythonBackend = await startBackend(logger);
  } catch (err) {
    logger.error('backend_startup_failed', { error: err.message });
    dialog.showErrorBox('MECH backend failed to start', err.message);
    app.exit(1);
    return;
  }

  registerIpcHandlers({ ipcMain, storage, logger });

  mainWindow = createMainWindow();
}

process.on('uncaughtException', (err) => {
  const logger = getLogger();
  logger.error('uncaught', { error: err.message });
});

app.whenReady().then(bootstrap).catch((err) => {
  console.error('Fatal during bootstrap:', err);
  app.exit(1);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', async () => {
  const logger = getLogger();
  await stopBackend(pythonBackend, logger);
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    mainWindow = createMainWindow();
  }
});

app.on('web-contents-created', (_event, contents) => {
  contents.on('will-navigate', (event, url) => {
    let allowed = false;
    try {
      const target = new URL(url);
      if (target.protocol === 'http:' || target.protocol === 'https:') {
        allowed = target.origin === new URL(RENDERER_DEV_URL).origin;
      } else if (target.protocol === 'file:') {
        const distRoot = path.join(__dirname, '..', 'dist') + path.sep;
        const fsPath = require('url').fileURLToPath(target);
        allowed = fsPath.startsWith(distRoot);
      }
    } catch {
      allowed = false;
    }
    if (!allowed) {
      event.preventDefault();
    }
  });
});
