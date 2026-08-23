'use strict';

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
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

  win.webContents.setWindowOpenHandler(({ url }) => {
    // Only allow http(s) external links to leave the app; block file:,
    // javascript:, data: and other schemes that openExternal would otherwise
    // honour.
    let safe = false;
    try {
      const parsed = new URL(url);
      safe = parsed.protocol === 'http:' || parsed.protocol === 'https:';
    } catch {
      safe = false;
    }
    if (safe) {
      shell.openExternal(url).catch(() => undefined);
    }
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
  if (process.env.VIRTUAL_ENV) {
    if (process.platform === 'win32') {
      candidates.push(path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe'));
      candidates.push(path.join(process.env.VIRTUAL_ENV, 'python.exe'));
    } else {
      candidates.push(path.join(process.env.VIRTUAL_ENV, 'bin', 'python'));
    }
  }

  const execDir = path.dirname(process.execPath || '');
  const searchRoots = [
    resourcesPath,
    execDir,
    __dirname,
    typeof app.getAppPath === 'function' ? app.getAppPath() : null,
    process.cwd()
  ].filter(Boolean);

  for (const root of searchRoots) {
    let curr = root;
    for (let depth = 0; depth < 8; depth++) {
      if (process.platform === 'win32') {
        candidates.push(path.join(curr, '.venv', 'Scripts', 'python.exe'));
        candidates.push(path.join(curr, 'venv', 'Scripts', 'python.exe'));
        candidates.push(path.join(curr, 'python', 'python.exe'));
        candidates.push(path.join(curr, 'python.exe'));
      } else {
        candidates.push(path.join(curr, '.venv', 'bin', 'python'));
        candidates.push(path.join(curr, 'venv', 'bin', 'python'));
        candidates.push(path.join(curr, 'python', 'bin', 'python'));
        candidates.push(path.join(curr, 'bin', 'python'));
      }
      const parent = path.dirname(curr);
      if (parent === curr) break;
      curr = parent;
    }
  }

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
  let base = app.isPackaged ? resourcesPath : repoRoot;

  let scriptPath = path.join(base, 'backend', SIDECAR_SCRIPT);
  if (!fs.existsSync(scriptPath)) {
    const execDir = path.dirname(process.execPath || '');
    const candidates = [
      path.join(resourcesPath, 'backend', SIDECAR_SCRIPT),
      path.join(execDir, 'resources', 'backend', SIDECAR_SCRIPT),
      path.join(__dirname, '..', '..', 'backend', SIDECAR_SCRIPT),
      path.join(process.cwd(), 'backend', SIDECAR_SCRIPT),
    ];
    for (const c of candidates) {
      if (fs.existsSync(c)) {
        scriptPath = c;
        base = path.dirname(path.dirname(c));
        break;
      }
    }
  }

  const pythonPath = resolvePythonPath(base);
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

  try {
    await bridge.start();
    logger.info('sidecar_ready');
  } catch (err) {
    logger.error('sidecar_start_failed', { error: err.message });
  }
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

  const backendUrl = process.env.VITE_BACKEND_URL || process.env.BACKEND_URL;
  const directHttpMode = !!backendUrl;

  let bridge;
  if (directHttpMode) {
    logger.info('direct_http_mode', { backendUrl });
    pythonBridge = null;
  } else {
    bridge = await startSidecar(logger);
    pythonBridge = bridge;
  }

  // Renderer-side services call window.desktopApi.httpRequest() for anything
  // that used to go to http://localhost:8000/api/*. Route it to the sidecar,
  // or fall back to an active HTTP backend server.
  ipcMain.handle('mech:http', async (_event, request) => {
    const payload = request || {};
    const headers = { ...(payload.headers || {}) };
    if (!headers['X-API-Key']) headers['X-API-Key'] = resolveApiKeyForRenderer();
    if (bridge && bridge.proc) {
      try {
        return await bridge.call('http', {
          method: payload.method || 'GET',
          path: payload.path || '/',
          headers,
          body: payload.body ?? '',
        });
      } catch (err) {
        logger.warn('bridge_http_failed_trying_fallback', { error: err.message });
      }
    }

    // Fallback: proxy directly to local HTTP server
    const backendUrl = process.env.VITE_BACKEND_URL || process.env.BACKEND_URL || 'http://127.0.0.1:8000';
    try {
      const http = require('node:http');
      const https = require('node:https');
      const url = new URL(backendUrl);
      const transport = url.protocol === 'https:' ? https : http;
      return await new Promise((resolve, reject) => {
        const reqPath = payload.path || '/';
        const reqMethod = payload.method || 'GET';
        const postData = typeof payload.body === 'string' ? payload.body : (payload.body ? JSON.stringify(payload.body) : '');
        if (postData && !headers['Content-Length']) {
          headers['Content-Length'] = Buffer.byteLength(postData);
        }
        const req = transport.request({
          hostname: url.hostname,
          port: url.port || (url.protocol === 'https:' ? '443' : '80'),
          path: reqPath,
          method: reqMethod,
          headers
        }, (res) => {
          let chunks = '';
          res.on('data', (c) => { chunks += c; });
          res.on('end', () => {
            resolve({
              status: res.statusCode || 200,
              statusText: res.statusMessage || '',
              headers: res.headers || {},
              body: chunks
            });
          });
        });
        req.on('error', (e) => reject(e));
        if (postData) req.write(postData);
        req.end();
      });
    } catch (fallbackErr) {
      throw new Error(`Failed to reach backend: ${fallbackErr.message}`);
    }
  });

  ipcMain.handle('mech:ping', async () => {
    if (bridge && bridge.proc) {
      try {
        return { ok: true, ...(await bridge.call('ping', {})) };
      } catch (err) {
        // try fallback
      }
    }
    const backendUrl = process.env.VITE_BACKEND_URL || process.env.BACKEND_URL || 'http://127.0.0.1:8000';
    try {
      const http = require('node:http');
      const https = require('node:https');
      const url = new URL(backendUrl);
      const transport = url.protocol === 'https:' ? https : http;
      return await new Promise((resolve) => {
        const req = transport.get(`${backendUrl}/health`, (res) => {
          resolve({ ok: res.statusCode === 200, status: 'running' });
        });
        req.on('error', (err) => resolve({ ok: false, error: err.message }));
      });
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
  // Restrict renderer navigations to either the dev server or the packaged
  // app's own frontend directory. The generic 'file://' prefix previously
  // permitted navigation to ANY local file (LFI risk); we now scope it to the
  // app's bundled frontend root.
  const appFileRoot = pathToFileURL(path.join(__dirname, '..')).href;
  contents.on('will-navigate', (event, url) => {
    if (url.startsWith(RENDERER_DEV_URL) || url.startsWith(appFileRoot)) {
      return;
    }
    event.preventDefault();
  });
});
