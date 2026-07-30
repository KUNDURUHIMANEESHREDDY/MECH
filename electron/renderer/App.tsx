import {
  Brain,
  Database,
  FolderOpen,
  HardDrive,
  Monitor,
  RefreshCw,
  Save,
  Settings as SettingsIcon,
  Terminal
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { getDesktopApi } from "./api";
import type {
  LogEntry,
  RecentFile,
  RecentProject,
  Settings,
  WorkspaceSummary
} from "../types";
import NeuronInspector from "./NeuronInspector";

const DEFAULT_SETTINGS: Settings = {
  theme: "system",
  gpuEnabled: false,
  cachePath: "",
  modelPath: "gpt2",
  workspacePath: ""
};

type ConnectionState = "connecting" | "connected" | "error";
type ViewMode = "settings" | "inspector";

function App() {
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [viewMode, setViewMode] = useState<ViewMode>("settings");
  const [connectionDetail, setConnectionDetail] = useState("Starting Python service");
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [projects, setProjects] = useState<RecentProject[]>([]);
  const [files, setFiles] = useState<RecentFile[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [workspace, setWorkspace] = useState<WorkspaceSummary | null>(null);
  const [status, setStatus] = useState("Ready");
  const [isSaving, setIsSaving] = useState(false);

  const api = useMemo(() => {
    try {
      return getDesktopApi();
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    if (!api) {
      setConnection("error");
      setConnectionDetail("Launch with Electron to enable Python IPC");
      return;
    }
    const desktopApi = api;

    let cancelled = false;
    const unsubscribe = desktopApi.onLogEntry((entry) => {
      setLogs((current) => [entry, ...current].slice(0, 200));
    });

    async function load() {
      try {
        const [ping, storedSettings, recentProjects, recentFiles, existingLogs] =
          await Promise.all([
            desktopApi.ping(),
            desktopApi.getSettings(),
            desktopApi.listRecentProjects(10),
            desktopApi.listRecentFiles(20),
            desktopApi.listLogs()
          ]);

        if (cancelled) {
          return;
        }

        setConnection("connected");
        setConnectionDetail(`Python connected: ${ping.storage}`);
        setSettings(storedSettings);
        setProjects(recentProjects);
        setFiles(recentFiles);
        setLogs(existingLogs.slice().reverse());

        if (storedSettings.workspacePath) {
          const summary = await desktopApi.describeWorkspace(storedSettings.workspacePath);
          if (!cancelled) {
            setWorkspace(summary);
          }
        }
      } catch (error) {
        if (!cancelled) {
          setConnection("error");
          setConnectionDetail(error instanceof Error ? error.message : String(error));
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [api]);

  useEffect(() => {
    document.documentElement.dataset.theme =
      settings.theme === "system" ? "" : settings.theme;
  }, [settings.theme]);

  async function refreshData() {
    if (!api) {
      return;
    }
    const [storedSettings, recentProjects, recentFiles] = await Promise.all([
      api.getSettings(),
      api.listRecentProjects(10),
      api.listRecentFiles(20)
    ]);
    setSettings(storedSettings);
    setProjects(recentProjects);
    setFiles(recentFiles);
    if (storedSettings.workspacePath) {
      setWorkspace(await api.describeWorkspace(storedSettings.workspacePath));
    }
    setStatus("Workspace state refreshed");
  }

  async function saveSettings(event: FormEvent) {
    event.preventDefault();
    if (!api) {
      return;
    }
    setIsSaving(true);
    try {
      const saved = await api.updateSettings(settings);
      setSettings(saved);
      if (saved.workspacePath) {
        setWorkspace(await api.describeWorkspace(saved.workspacePath));
      }
      setStatus("Settings saved");
    } finally {
      setIsSaving(false);
    }
  }

  async function chooseProject() {
    if (!api) {
      return;
    }
    const project = await api.chooseProject();
    if (project) {
      const nextSettings = {
        ...settings,
        workspacePath: project.path
      };
      setSettings(await api.updateSettings(nextSettings));
      setProjects(await api.listRecentProjects(10));
      setWorkspace(await api.describeWorkspace(project.path));
      setStatus(`Opened ${project.name}`);
    }
  }

  async function registerWorkspace() {
    if (!api || !settings.workspacePath.trim()) {
      return;
    }
    const project = await api.addRecentProject(settings.workspacePath);
    setProjects(await api.listRecentProjects(10));
    setWorkspace(await api.describeWorkspace(project.path));
    setStatus(`Registered ${project.name}`);
  }

  async function chooseCachePath() {
    if (!api) {
      return;
    }
    const selectedPath = await api.chooseCachePath();
    if (selectedPath) {
      setSettings((current) => ({ ...current, cachePath: selectedPath }));
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Desktop runtime</p>
          <h1>Neural Debugger</h1>
        </div>
        <div className={`connection connection-${connection}`}>
          <Monitor size={18} aria-hidden="true" />
          <span>{connection === "connected" ? "Python connected" : connection}</span>
        </div>
      </header>

      <section className="workspace-strip" aria-label="Workspace status">
        <div>
          <span className="label">Workspace</span>
          <strong>{workspace?.name ?? "No project selected"}</strong>
          <span className="muted">{workspace?.path ?? settings.workspacePath}</span>
        </div>
        <div>
          <span className="label">Files indexed</span>
          <strong>{workspace?.fileCount ?? 0}</strong>
        </div>
        <div>
          <span className="label">Model</span>
          <strong>{settings.modelPath}</strong>
        </div>
      </section>

      <div className="main-grid">
        <aside className="panel explorer-panel" aria-label="Workspace explorer">
          <div className="panel-header">
            <h2>Projects</h2>
            <button className="icon-button" type="button" onClick={refreshData} aria-label="Refresh">
              <RefreshCw size={16} aria-hidden="true" />
            </button>
          </div>

          <button className="primary-button" type="button" onClick={chooseProject}>
            <FolderOpen size={16} aria-hidden="true" />
            Open Project
          </button>

          <div className="list" aria-label="Recent projects">
            {projects.length === 0 ? (
              <p className="empty-state">Recent projects will appear here.</p>
            ) : (
              projects.map((project) => (
                <button
                  className="list-item"
                  type="button"
                  key={project.id}
                  onClick={() =>
                    setSettings((current) => ({
                      ...current,
                      workspacePath: project.path
                    }))
                  }
                >
                  <span>{project.name}</span>
                  <small>{project.path}</small>
                </button>
              ))
            )}
          </div>
        </aside>

        <section className="panel settings-panel">
          <div className="panel-header">
            <div className="view-tabs">
              <button
                className={`view-tab ${viewMode === "settings" ? "active" : ""}`}
                onClick={() => setViewMode("settings")}
              >
                <SettingsIcon size={16} aria-hidden="true" />
                Settings
              </button>
              <button
                className={`view-tab ${viewMode === "inspector" ? "active" : ""}`}
                onClick={() => setViewMode("inspector")}
              >
                <Brain size={16} aria-hidden="true" />
                Inspector
              </button>
            </div>
          </div>

          {viewMode === "inspector" ? (
            <NeuronInspector />
          ) : (
          <form className="settings-form" onSubmit={saveSettings}>
            <label>
              <span>Theme</span>
              <select
                value={settings.theme}
                onChange={(event) =>
                  setSettings((current) => ({
                    ...current,
                    theme: event.target.value as Settings["theme"]
                  }))
                }
              >
                <option value="system">System</option>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
            </label>

            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={settings.gpuEnabled}
                onChange={(event) =>
                  setSettings((current) => ({
                    ...current,
                    gpuEnabled: event.target.checked
                  }))
                }
              />
              <span>Enable GPU execution when available</span>
            </label>

            <label>
              <span>Model path</span>
              <input
                value={settings.modelPath}
                onChange={(event) =>
                  setSettings((current) => ({
                    ...current,
                    modelPath: event.target.value
                  }))
                }
                placeholder="gpt2 or local model path"
              />
            </label>

            <label>
              <span>Workspace path</span>
              <div className="input-row">
                <input
                  value={settings.workspacePath}
                  onChange={(event) =>
                    setSettings((current) => ({
                      ...current,
                      workspacePath: event.target.value
                    }))
                  }
                  placeholder="Project folder"
                />
                <button type="button" onClick={registerWorkspace}>
                  <FolderOpen size={16} aria-hidden="true" />
                  Register
                </button>
              </div>
            </label>

            <label>
              <span>Cache path</span>
              <div className="input-row">
                <input
                  value={settings.cachePath}
                  onChange={(event) =>
                    setSettings((current) => ({
                      ...current,
                      cachePath: event.target.value
                    }))
                  }
                  placeholder="Activation and model cache"
                />
                <button type="button" onClick={chooseCachePath}>
                  <HardDrive size={16} aria-hidden="true" />
                  Browse
                </button>
              </div>
            </label>

            <button className="primary-button" type="submit" disabled={isSaving}>
              <Save size={16} aria-hidden="true" />
              {isSaving ? "Saving" : "Save Settings"}
            </button>
          </form>
          )}
        </section>

        <section className="panel recent-panel" aria-labelledby="recent-title">
          <div className="panel-header">
            <h2 id="recent-title">Recent Files</h2>
            <Database size={18} aria-hidden="true" />
          </div>
          <div className="list">
            {files.length === 0 ? (
              <p className="empty-state">Recent files will appear after project activity.</p>
            ) : (
              files.map((file) => (
                <div className="file-row" key={file.id}>
                  <span>{file.path}</span>
                  <small>{file.projectPath ?? "No project"}</small>
                </div>
              ))
            )}
          </div>
        </section>

        <section className="panel log-panel" aria-labelledby="logs-title">
          <div className="panel-header">
            <h2 id="logs-title">Runtime Logs</h2>
            <Terminal size={18} aria-hidden="true" />
          </div>
          <div className="log-list" role="log" aria-live="polite">
            {logs.length === 0 ? (
              <p className="empty-state">Runtime events will appear here.</p>
            ) : (
              logs.map((entry, index) => (
                <div className={`log-entry log-${entry.level}`} key={`${entry.timestamp}-${index}`}>
                  <time>{new Date(entry.timestamp).toLocaleTimeString()}</time>
                  <span>{entry.level}</span>
                  <p>{entry.message}</p>
                </div>
              ))
            )}
          </div>
        </section>
      </div>

      <footer className="statusbar">
        <span>{status}</span>
        <span>{connectionDetail}</span>
      </footer>
    </main>
  );
}

export default App;
