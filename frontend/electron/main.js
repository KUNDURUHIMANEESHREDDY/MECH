'use strict';

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('node:path');
const { spawn } = require('node:child_process');
const http = require('node:http');
const { registerIpcHandlers } = require('./ipc');
const { getLogger } = require('./logger');
const { getStorage } = require('./storage');

const isDev = !app.isPackaged;
const RENDERER_DEV_URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';
const BACKEND_PORT = 8000;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;

let mainWindow = null;
let pythonBackend = null;

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

async function startBackend(logger) {
  try {
    const res = await new Promise((resolve, reject) => {
      const req = http.get(`${BACKEND_URL}/health`, (res) => { res.resume(); resolve(res); });
      req.on('error', reject);
      req.setTimeout(2000, () => { req.destroy(); reject(new Error('timeout')); });
    });
    if (res.statusCode && res.statusCode < 500) {
      logger.info('backend_reuse', { url: BACKEND_URL });
      return null;
    }
  } catch {
  }

  let pythonPath, scriptPath, cwd, pythonPathEnv;

  if (isDev) {
    pythonPath = 'python';
    scriptPath = path.join(__dirname, '..', '..', 'backend', 'main.py');
    cwd = path.join(__dirname, '..', '..');
    pythonPathEnv = `${cwd};${path.join(cwd, 'backend')}`;
  } else {
    const resourcesPath = process.resourcesPath || path.join(__dirname, '..');
    pythonPath = 'python';
    scriptPath = path.join(resourcesPath, 'backend', 'main.py');
    cwd = resourcesPath;
    pythonPathEnv = `${resourcesPath};${path.join(resourcesPath, 'backend')}`;
  }

  logger.info('backend_spawning', { pythonPath, scriptPath, cwd, pythonPathEnv });

  const child = spawn(pythonPath, [scriptPath], {
    cwd,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONPATH: pythonPathEnv, PYTHONUNBUFFERED: '1' },
    windowsHide: true,
  });

  child.stdout?.on('data', (d) => logger.info('backend_stdout', { line: d.toString().trim() }));
  child.stderr?.on('data', (d) => logger.warn('backend_stderr', { line: d.toString().trim() }));
  child.on('exit', (code) => logger.info('backend_exited', { code }));
  child.on('error', (err) => logger.error('backend_spawn_error', { error: err.message }));

  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    try {
      const res = await new Promise((resolve, reject) => {
        const req = http.get(`${BACKEND_URL}/health`, (res) => { res.resume(); resolve(res); });
        req.on('error', reject);
        req.setTimeout(2000, () => { req.destroy(); reject(new Error('timeout')); });
      });
      if (res.statusCode && res.statusCode < 500) {
        logger.info('backend_ready', { url: BACKEND_URL });
        return child;
      }
    } catch { }
    await new Promise((r) => setTimeout(r, 500));
  }

  logger.warn('backend_not_ready', { timeout: '30s' });
  return child;
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

  const storage = getStorage();
  await storage.init();
  logger.info('storage_ready', { db: storage.dbPath });

  pythonBackend = await startBackend(logger);
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
    const allowed = [RENDERER_DEV_URL, 'file://'];
    if (!allowed.some((prefix) => url.startsWith(prefix))) {
      event.preventDefault();
    }
  });
});
