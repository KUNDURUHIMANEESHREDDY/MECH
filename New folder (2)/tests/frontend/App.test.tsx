import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";

import App from "../../electron/renderer/App";
import type { DesktopApi, Settings } from "../../electron/types";

const settings: Settings = {
  theme: "system",
  gpuEnabled: false,
  cachePath: "C:/cache",
  modelPath: "gpt2",
  workspacePath: "C:/project"
};

const stubInspect = vi.fn().mockResolvedValue({ neuron_id: "", layer: "", layer_index: 0, neuron_index: 0, activation: 0, statistics: { max: 0, mean: 0, variance: 0, sparsity: 0 } });
const stubList = vi.fn().mockResolvedValue([]);

function makeApi(overrides: Partial<DesktopApi> = {}): DesktopApi {
  return {
    ping: vi.fn().mockResolvedValue({ ok: true, storage: "C:/state/desktop.sqlite3" }),
    getSettings: vi.fn().mockResolvedValue(settings),
    updateSettings: vi.fn().mockImplementation((next: Settings) => Promise.resolve(next)),
    listRecentProjects: vi.fn().mockResolvedValue([]),
    addRecentProject: vi.fn().mockResolvedValue({
      id: 1,
      path: "C:/project",
      name: "project",
      openedAt: "2026-07-25T00:00:00Z"
    }),
    chooseProject: vi.fn().mockResolvedValue(null),
    listRecentFiles: vi.fn().mockResolvedValue([]),
    addRecentFile: vi.fn(),
    describeWorkspace: vi.fn().mockResolvedValue({
      path: "C:/project",
      exists: true,
      name: "project",
      fileCount: 12
    }),
    chooseCachePath: vi.fn().mockResolvedValue("C:/cache"),
    listLogs: vi.fn().mockResolvedValue([]),
    onLogEntry: vi.fn().mockReturnValue(() => undefined),
    modelInfo: vi.fn().mockResolvedValue({ num_layers: 12, num_heads: 12, hidden_dim: 768, seq_len: 16, layer_names: [] }),
    prompt: { run: stubInspect, tokenLookup: stubInspect, ioi: stubInspect, cacheShapes: stubInspect, ablate: stubInspect, multiAblate: stubInspect, attentionPattern: stubInspect, patchingMatrix: stubInspect, logitLens: stubInspect as any, neuronInspect: stubInspect as any, neuronSearch: stubInspect as any, correlatedNeurons: stubInspect as any, neuronEvolution: stubInspect as any, datasetActivation: stubInspect as any, predictionTrace: stubInspect as any, circuitTrace: stubInspect as any, promptCompare: stubInspect as any, runExperiment: stubInspect as any },
    ...overrides
  };
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("App", () => {
  test("connects to Python and renders persisted settings", async () => {
    Object.defineProperty(window, "desktopApi", {
      value: makeApi(),
      configurable: true
    });

    render(<App />);

    expect(await screen.findByText("Python connected")).toBeInTheDocument();
    expect(await screen.findByDisplayValue("gpt2")).toBeInTheDocument();
    expect(await screen.findByText("12")).toBeInTheDocument();
  });

  test("saves changed settings through IPC", async () => {
    const api = makeApi();
    Object.defineProperty(window, "desktopApi", {
      value: api,
      configurable: true
    });

    render(<App />);

    const themeSelect = await screen.findByLabelText("Theme");
    fireEvent.change(themeSelect, { target: { value: "dark" } });
    fireEvent.click(screen.getByRole("button", { name: /save settings/i }));

    await waitFor(() => {
      expect(api.updateSettings).toHaveBeenCalledWith(
        expect.objectContaining({ theme: "dark" })
      );
    });
    expect(await screen.findByText("Settings saved")).toBeInTheDocument();
  });
});
