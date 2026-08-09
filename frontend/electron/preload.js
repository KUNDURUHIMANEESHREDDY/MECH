'use strict';

const { contextBridge, ipcRenderer } = require('electron');

// Expose a safe, minimal API to the renderer via contextBridge.
contextBridge.exposeInMainWorld('appApi', {
  getSettings: () => ipcRenderer.invoke('settings:get'),
  setSettings: (patch) => ipcRenderer.invoke('settings:set', patch),
  resetSettings: () => ipcRenderer.invoke('settings:reset'),

  // Workspaces & recent sessions
  listProjects: () => ipcRenderer.invoke('projects:list'),
  addProject: (project) => ipcRenderer.invoke('projects:add', project),
  removeProject: (id) => ipcRenderer.invoke('projects:remove', id),
  listRecentFiles: () => ipcRenderer.invoke('recent:list'),
  addRecentFile: (file) => ipcRenderer.invoke('recent:add', file),
  clearRecentFiles: () => ipcRenderer.invoke('recent:clear'),

  // Sessions & Experiments (Neural Debugger domain)
  listSessions: () => ipcRenderer.invoke('sessions:list'),
  saveSession: (session) => ipcRenderer.invoke('sessions:upsert', session),
  removeSession: (id) => ipcRenderer.invoke('sessions:remove', id),
  listExperiments: () => ipcRenderer.invoke('experiments:list'),
  saveExperiment: (exp) => ipcRenderer.invoke('experiments:upsert', exp),
  removeExperiment: (id) => ipcRenderer.invoke('experiments:remove', id),

  // Python bridge & Runtime Engine
  pythonPing: () => ipcRenderer.invoke('python:ping'),
  pythonCall: (method, payload) => ipcRenderer.invoke('python:call', { method, payload }),
  getRuntimeStatus: () => ipcRenderer.invoke('runtime:status'),
  analyzeTokens: (prompt) => ipcRenderer.invoke('runtime:analyzeTokens', prompt),

  // GPT-2 live interpretability pipeline
  gpt2Load:          ()       => ipcRenderer.invoke('gpt2:load', {}),
  gpt2RunPrompt:     (prompt) => ipcRenderer.invoke('gpt2:runPrompt', { prompt }),
  gpt2GetActivations:(layer)  => ipcRenderer.invoke('gpt2:activations', { layer }),
  gpt2AttentionHead: (layer, head) => ipcRenderer.invoke('gpt2:attentionHead', { layer, head }),
  gpt2PatchHead:     (layer, head, posToken, negToken) =>
    ipcRenderer.invoke('gpt2:patchHead', { layer, head, pos_token: posToken, neg_token: negToken }),
  gpt2RunIoi:        (ioName, subjName) =>
    ipcRenderer.invoke('gpt2:ioi', { io_name: ioName, subj_name: subjName }),

  // Build & logging
  startBuild: (options) => ipcRenderer.invoke('build:start', options || {}),
  getBuildLogs: () => ipcRenderer.invoke('build:logs'),
  clearBuildLogs: () => ipcRenderer.invoke('build:clear'),
  onBuildEvent: (handler) => {
    const wrapped = (_event, data) => handler(data);
    ipcRenderer.on('build:event', wrapped);
    return () => ipcRenderer.removeListener('build:event', wrapped);
  },
  getAppLogs: () => ipcRenderer.invoke('logs:get'),

  // System
  openExternal: (url) => ipcRenderer.invoke('system:openExternal', url),
  showInFolder: (p) => ipcRenderer.invoke('system:showInFolder', p)
});

// API-key bridge + serverless fetch transport used by src/services/*.
// httpRequest() executes any "http://localhost:8000/..." renderer fetch against
// the in-process sidecar; the main process maps it to the PythonBridge 'http' call.
contextBridge.exposeInMainWorld('desktopApi', {
  getApiKey: () => ipcRenderer.invoke('api-key:get'),
  httpRequest: (request) => ipcRenderer.invoke('mech:http', request),
  ping: () => ipcRenderer.invoke('mech:ping')
});
