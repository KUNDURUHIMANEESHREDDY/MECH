import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { mkdirSync } from "node:fs";
import path from "node:path";

import { CHANNELS } from "./ipc/channels";
import { PythonBridge } from "./ipc/pythonBridge";
import { DesktopLogger } from "./logging";

let mainWindow: BrowserWindow | null = null;
let pythonBridge: PythonBridge | null = null;
let logger: DesktopLogger | null = null;

const configuredUserData = process.env.NEURAL_DEBUGGER_USER_DATA;
if (configuredUserData) {
  app.setPath("userData", configuredUserData);
}

function getAppRoot(): string {
  if (app.isPackaged) {
    return process.resourcesPath;
  }
  return path.resolve(__dirname, "..");
}

function createPythonBridge(): PythonBridge {
  const userDataPath = app.getPath("userData");
  mkdirSync(path.join(userDataPath, "storage"), { recursive: true });

  const desktopLogger = new DesktopLogger(userDataPath);
  logger = desktopLogger;

  const bridge = new PythonBridge({
    pythonCommand: process.env.NEURAL_DEBUGGER_PYTHON ?? "python",
    scriptPath: path.join(getAppRoot(), "scripts", "desktop_service.py"),
    dbPath:
      process.env.NEURAL_DEBUGGER_DB_PATH ??
      path.join(userDataPath, "storage", "desktop.sqlite3"),
    logger: desktopLogger
  });
  pythonBridge = bridge;
  return bridge;
}

function registerIpcHandlers(bridge: PythonBridge, desktopLogger: DesktopLogger): void {
  ipcMain.handle(
    CHANNELS.pythonRequest,
    async (_event, method: string, params: Record<string, unknown> = {}) => {
      desktopLogger.info("Renderer IPC request", { method });
      return bridge.request(method, params);
    }
  );

  ipcMain.handle(CHANNELS.logsList, () => desktopLogger.getEntries());

  ipcMain.handle(CHANNELS.workspaceChooseCachePath, async () => {
    const result = await dialog.showOpenDialog({
      title: "Choose cache folder",
      properties: ["openDirectory", "createDirectory"]
    });
    return result.canceled ? null : result.filePaths[0];
  });

  ipcMain.handle(CHANNELS.workspaceChooseProject, async () => {
    const result = await dialog.showOpenDialog({
      title: "Open project",
      properties: ["openDirectory"]
    });
    if (result.canceled || result.filePaths.length === 0) {
      return null;
    }

    const projectPath = result.filePaths[0];
    return bridge.request("projects.addRecent", {
      path: projectPath,
      name: path.basename(projectPath)
    });
  });
}

async function createWindow(): Promise<void> {
  const window = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: "#101418",
    title: "Neural Debugger",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });
  mainWindow = window;
  logger?.setBroadcaster((entry) => {
    if (!window.isDestroyed()) {
      window.webContents.send(CHANNELS.logsEntry, entry);
    }
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    await window.loadURL(process.env.VITE_DEV_SERVER_URL);
    window.webContents.openDevTools({ mode: "detach" });
  } else {
    await window.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }
}

app.whenReady().then(async () => {
  const bridge = createPythonBridge();
  const desktopLogger = logger;
  if (!desktopLogger) {
    throw new Error("Desktop logger failed to initialize.");
  }

  registerIpcHandlers(bridge, desktopLogger);
  await createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createWindow();
    }
  });
});

app.on("before-quit", () => {
  pythonBridge?.stop();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
