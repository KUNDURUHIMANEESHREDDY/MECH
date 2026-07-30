'use strict';

const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path = require('node:path');
const { registerIpcHandlers } = require('./ipc');
const { getLogger } = require('./logger');
const { getStorage } = require('./storage');
const { getPythonBridge } = require('./python');

const isDev = !app.isPackaged;
const RENDERER_DEV_URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';

let mainWindow = null;
let pythonBridge = null;

function createMainWindow() {
  const logger = getLogger();

  const win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 600,
    show: false,
    backgroundColor: '#0e1116',
    title: 'DesktopApp',
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

async function bootstrap() {
  const logger = getLogger();
  logger.info('app_boot', { version: app.getVersion(), isDev });

  // Initialize storage (creates DB on first run)
  const storage = getStorage();
  await storage.init();
  logger.info('storage_ready', { db: storage.dbPath });

  // Start Python bridge
  pythonBridge = getPythonBridge({ logger });
  try {
    await pythonBridge.start();
    logger.info('python_connected', { pid: pythonBridge.pid });
  } catch (err) {
    logger.error('python_connect_failed', { error: err.message });
  }

  // Register IPC handlers
  registerIpcHandlers({ ipcMain, storage, pythonBridge, logger });

  // Create window
  mainWindow = createMainWindow();
}

app.whenReady().then(bootstrap).catch((err) => {
  // eslint-disable-next-line no-console
  console.error('Fatal during bootstrap:', err);
  app.exit(1);
});

app.on('window-all-closed', async () => {
  if (pythonBridge) {
    await pythonBridge.stop();
    pythonBridge = null;
  }
  if (process.platform !== 'darwin') app.quit();
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
