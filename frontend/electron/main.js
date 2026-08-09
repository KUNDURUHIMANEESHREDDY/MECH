'use strict';

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('node:path');
const crypto = require('node:crypto');
const fs = require('node:fs');
const { registerIpcHandlers } = require('./ipc');
const { getLogger } = require('./logger');
const { getStorage } = require('./storage');
const { getPythonBridge } = require('./python');

const isDev = !app.isPackaged;
const RENDERER_DEV_URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';
const SIDECAR_SCRIPT = 'mech_service.py';

let mainWindow = null;
let pythonBridge = null;
let apiKey = null;

// Renderer (src/services/api.ts) reads the API key via preload's
// desktopApi.getApiKey() and sends it as the X-API-Key header. We generate it
// once in userData and hand the same value to the spawned backend via
// MECH_API_KEY so both sides agree.
function getApiKeyFilePath() {
  return path.join(app.getPath('userData'), 'storage', 'api_key.txt');
}

function ensureApiKey() {
  if (apiKey) return apiKey;
  const keyFile = getApiKeyFilePath();
  try {
    const existing = fs.readFileSync(keyFile, 'utf-8').trim();
    if (existing) {
      apiKey = existing;
      return apiKey;
    }
  } catch { /* not persisted yet */ }
  const fresh = crypto.randomBytes(32).toString('base64url');
  try {
    fs.mkdirSync(path.dirname(keyFile), { recursive: true });
    fs.writeFileSync(keyFile, fresh, { encoding: 'utf-8', mode: 0o600 });
  } catch { /* non-fatal: key still valid for this session */ }
  apiKey = fresh;
  return apiKey;
}

function getBackendKeyFilePath() {
  // Where a detached/manual backend writes its key:
  // dev: <repoRoot>/storage/api_key.txt, packaged: <resources>/storage/api_key.txt
  if (app.isPackaged) {
    return path.join(process.resourcesPath || path.join(__dirname, '..'), 'storage', 'api_key.txt');
  }
  return path.join(__dirname, '..', '..', 'storage', 'api_key.txt');
}

function resolveApiKeyForRenderer() {
  // 1. Key we generated (userData) — matches MECH_API_KEY when we spawned.
  try {
    const k = fs.readFileSync(getApiKeyFilePath(), 'utf-8').trim();
    if (k) return k;
  } catch { }
  // 2. Key written by a backend we reused (started detached / manually).
  try {
    const k = fs.readFileSync(getBackendKeyFilePath(), 'utf-8').trim();
    if (k) return k;
  } catch { }
  return '';
}

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
      sandbox: false,
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

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url).catch(() => undefined);
    return { action: 'deny' };
  });

  if (process.env.VITE_DEV_SERVER_URL) {
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

function resolvePythonPath(resourcesPath) {
  const candidates = [];
  if (process.env.MECH_PYTHON) candidates.push(process.env.MECH_PYTHON);
  if (process.env.PYTHON_PATH) candidates.push(process.env.PYTHON_PATH);

  // Use bundled/local Python runtimes when present.
  candidates.push(path.join(resourcesPath, 'python', 'python.exe'));
  candidates.push(path.join(resourcesPath, '.venv', 'Scripts', 'python.exe'));

  // Development / local checkout: use the project's virtual environment.
  const repoRoot = path.join(__dirname, '..', '..');
  candidates.push(path.join(repoRoot, '.venv', 'Scripts', 'python.exe'));
  candidates.push(path.join(repoRoot, 'venv', 'Scripts', 'python.exe'));

  for (const candidate of candidates) {
    try {
      if (candidate && fs.existsSync(candidate)) return candidate;
    } catch { }
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

async function startSidecar(logger) {
  const repoRoot = path.join(__dirname, '..', '..');
  const resourcesPath = process.resourcesPath || path.join(__dirname, '..');
  const base = app.isPackaged ? resourcesPath : repoRoot;

  const pythonPath = resolvePythonPath(base);
  const scriptPath = path.join(base, 'backend', SIDECAR_SCRIPT);
  const pythonPathEnv = `${base};${path.join(base, 'backend')}`;
  const storageDb = path.join(base, 'backend', 'storage', 'mech.db');

  logger.info('sidecar_spawning', { pythonPath, scriptPath, base });

  const bridge = getPythonBridge({
    logger,
    pythonPath,
    scriptPath,
    pythonPathEnv,
    cwd: base,
    env: {
      MECH_API_KEY: ensureApiKey(),
      MECH_STORAGE_DB: storageDb,
    },
  });

  await bridge.start();
  logger.info('sidecar_ready');
  return bridge;
}

async function stopSidecar(bridge, logger) {
  if (!bridge) return;
  try {
    await bridge.stop();
  } catch (e) {
    logger.error('sidecar_stop_error', { error: e.message });
  }
}

async function bootstrap() {
  const logger = getLogger();
  logger.info('app_boot', { version: app.getVersion(), isDev });

  const storage = getStorage();
  await storage.init();
  logger.info('storage_ready', { db: storage.dbPath });

  const bridge = await startSidecar(logger);
  pythonBridge = bridge;

  // Renderer-side services call window.desktopApi.httpRequest() for anything
  // that used to go to http://localhost:8000/api/*. Route it to the sidecar.
  ipcMain.handle('mech:http', async (_event, request) => {
    const payload = request || {};
    const headers = { ...(payload.headers || {}) };
    if (!headers['X-API-Key']) headers['X-API-Key'] = resolveApiKeyForRenderer();
    return bridge.call('http', {
      method: payload.method || 'GET',
      path: payload.path || '/',
      headers,
      body: payload.body ?? '',
    });
  });

  ipcMain.handle('mech:ping', async () => {
    try {
      return { ok: true, ...(await bridge.call('ping', {})) };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  });

  registerIpcHandlers({ ipcMain, storage, pythonBridge: bridge, logger, getApiKey: resolveApiKeyForRenderer });
  ipcMain.handle('api-key:get', () => resolveApiKeyForRenderer());

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
  await stopSidecar(pythonBridge, logger);
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    mainWindow = createMainWindow();
  }
});

app.on('web-contents-created', (_event, contents) => {
  contents.on('will-navigate', (event, url) => {
    const allowed = [RENDERER_DEV_URL, 'file://'];
    if (!allowed.some((prefix) => url.startsWith(prefix))) {
      event.preventDefault();
    }
  });
});
