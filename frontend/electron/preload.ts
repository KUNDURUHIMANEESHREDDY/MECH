import { contextBridge, ipcRenderer } from "electron";

import { CHANNELS } from "./ipc/channels";
import type { DesktopApi, LogEntry, Settings } from "./types";

function pythonRequest(method: string, params: Record<string, unknown> = {}) {
  return ipcRenderer.invoke(CHANNELS.pythonRequest, method, params);
}

const api: DesktopApi = {
  ping: () => pythonRequest("ping", {}),
  getApiKey: () => ipcRenderer.invoke("mech:get-api-key"),
  getSettings: () => pythonRequest("settings.get", {}),
  updateSettings: (settings: Settings) =>
    pythonRequest("settings.update", { settings }),
  listRecentProjects: (limit = 10) =>
    pythonRequest("projects.list", { limit }),
  addRecentProject: (projectPath: string, name?: string) =>
    pythonRequest("projects.addRecent", { path: projectPath, name }),
  chooseProject: () => ipcRenderer.invoke(CHANNELS.workspaceChooseProject),
  listRecentFiles: (limit = 20) =>
    pythonRequest("recentFiles.list", { limit }),
  addRecentFile: (filePath: string, projectPath?: string) =>
    pythonRequest("recentFiles.add", { path: filePath, projectPath }),
  describeWorkspace: (workspacePath: string) =>
    pythonRequest("workspace.describe", { path: workspacePath }),
  chooseCachePath: () => ipcRenderer.invoke(CHANNELS.workspaceChooseCachePath),
  listLogs: () => ipcRenderer.invoke(CHANNELS.logsList),
  onLogEntry: (callback: (entry: LogEntry) => void) => {
    const listener = (_event: Electron.IpcRendererEvent, entry: LogEntry) => {
      callback(entry);
    };
    ipcRenderer.on(CHANNELS.logsEntry, listener);
    return () => ipcRenderer.removeListener(CHANNELS.logsEntry, listener);
  },

  modelInfo: () => pythonRequest("neuron.modelInfo", {}),

  prompt: {
    run: (prompt: string, topK = 20) =>
      pythonRequest("prompt.run", { prompt, top_k: topK }),
    tokenLookup: (token: string) =>
      pythonRequest("prompt.tokenLookup", { token }),
    ioi: () =>
      pythonRequest("prompt.ioi", {}),
    cacheShapes: (prompt = "The capital of France is") =>
      pythonRequest("prompt.cacheShapes", { prompt }),
    ablate: (prompt: string, layer: number, head: number) =>
      pythonRequest("prompt.ablate", { prompt, layer, head }),
    multiAblate: (prompt = "When Mary and John went to the store, John gave the bag to", heads = [[9, 9], [9, 6], [10, 0]]) =>
      pythonRequest("prompt.multiAblate", { prompt, heads }),
    attentionPattern: (prompt: string, layer: number, head: number) =>
      pythonRequest("prompt.attentionPattern", { prompt, layer, head }),
    patchingMatrix: (prompt = "When Mary and John went to the store, John gave the bag to") =>
      pythonRequest("prompt.patchingMatrix", { prompt }),
    logitLens: (prompt = "The capital of France is", topK = 5, targetToken?: string) =>
      pythonRequest("prompt.logitLens", { prompt, top_k: topK, target_token: targetToken }),
    neuronInspect: (prompt: string, layer: number, neuron: number, targetToken?: string) =>
      pythonRequest("prompt.neuronInspect", { prompt, layer, neuron, target_token: targetToken }),
    neuronSearch: (prompt: string, layer: number, topK = 20) =>
      pythonRequest("prompt.neuronSearch", { prompt, layer, top_k: topK }),
    correlatedNeurons: (prompt: string, layer: number, neuron: number, topK = 20) =>
      pythonRequest("prompt.correlatedNeurons", { prompt, layer, neuron, top_k: topK }),
    neuronEvolution: (prompt: string, layer: number, neuron: number, tokenIndex: number) =>
      pythonRequest("prompt.neuronEvolution", { prompt, layer, neuron, token_index: tokenIndex }),
    datasetActivation: (layer: number, neuron: number, prompts?: string[], topK = 50) =>
      pythonRequest("prompt.datasetActivation", { layer, neuron, prompts, top_k: topK }),
    predictionTrace: (prompt: string, targetToken?: string, layerForHeads?: number) =>
      pythonRequest("prompt.predictionTrace", { prompt, target_token: targetToken, layer_for_heads: layerForHeads }),
    circuitTrace: (prompt: string, targetToken?: string) =>
      pythonRequest("prompt.circuitTrace", { prompt, targetToken }),
    promptCompare: (promptA: string, promptB: string, topK = 10) =>
      pythonRequest("prompt.promptCompare", { prompt_a: promptA, prompt_b: promptB, top_k: topK }),
    runExperiment: (prompts: string[], config?) =>
      pythonRequest("prompt.runExperiment", { prompts, config: config ?? {} }),
  },

};
    ipcRenderer.on(CHANNELS.logsEntry, listener);
    return () => ipcRenderer.removeListener(CHANNELS.logsEntry, listener);
  },

  // Neuron Inspector API
  modelInfo: () => pythonRequest("neuron.modelInfo", {}),

  prompt: {
    run: (prompt: string, topK = 20) =>
      pythonRequest("prompt.run", { prompt, top_k: topK }),
    tokenLookup: (token: string) =>
      pythonRequest("prompt.tokenLookup", { token }),
    ioi: () =>
      pythonRequest("prompt.ioi", {}),
    cacheShapes: (prompt = "The capital of France is") =>
      pythonRequest("prompt.cacheShapes", { prompt }),
    ablate: (prompt: string, layer: number, head: number) =>
      pythonRequest("prompt.ablate", { prompt, layer, head }),
    multiAblate: (prompt = "When Mary and John went to the store, John gave the bag to", heads = [[9, 9], [9, 6], [10, 0]]) =>
      pythonRequest("prompt.multiAblate", { prompt, heads }),
    attentionPattern: (prompt: string, layer: number, head: number) =>
      pythonRequest("prompt.attentionPattern", { prompt, layer, head }),
    patchingMatrix: (prompt = "When Mary and John went to the store, John gave the bag to") =>
      pythonRequest("prompt.patchingMatrix", { prompt }),
    logitLens: (prompt = "The capital of France is", topK = 5, targetToken?: string) =>
      pythonRequest("prompt.logitLens", { prompt, top_k: topK, target_token: targetToken }),
    neuronInspect: (prompt: string, layer: number, neuron: number, targetToken?: string) =>
      pythonRequest("prompt.neuronInspect", { prompt, layer, neuron, target_token: targetToken }),
    neuronSearch: (prompt: string, layer: number, topK = 20) =>
      pythonRequest("prompt.neuronSearch", { prompt, layer, top_k: topK }),
    correlatedNeurons: (prompt: string, layer: number, neuron: number, topK = 20) =>
      pythonRequest("prompt.correlatedNeurons", { prompt, layer, neuron, top_k: topK }),
    neuronEvolution: (prompt: string, layer: number, neuron: number, tokenIndex: number) =>
      pythonRequest("prompt.neuronEvolution", { prompt, layer, neuron, token_index: tokenIndex }),
    datasetActivation: (layer: number, neuron: number, prompts?: string[], topK = 50) =>
      pythonRequest("prompt.datasetActivation", { layer, neuron, prompts, top_k: topK }),
    predictionTrace: (prompt: string, targetToken?: string, layerForHeads?: number) =>
      pythonRequest("prompt.predictionTrace", { prompt, target_token: targetToken, layer_for_heads: layerForHeads }),
    circuitTrace: (prompt: string, targetToken?: string) =>
      pythonRequest("prompt.circuitTrace", { prompt, target_token: targetToken }),
    promptCompare: (promptA: string, promptB: string, topK = 10) =>
      pythonRequest("prompt.promptCompare", { prompt_a: promptA, prompt_b: promptB, top_k: topK }),
    runExperiment: (prompts: string[], config?) =>
      pythonRequest("prompt.runExperiment", { prompts, config: config ?? {} }),
  },

};

contextBridge.exposeInMainWorld("desktopApi", api);

declare global {
  interface Window {
    desktopApi: DesktopApi;
  }
}
