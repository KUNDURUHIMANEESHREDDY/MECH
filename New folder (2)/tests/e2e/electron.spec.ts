import { _electron as electron, test, expect } from "@playwright/test";
import path from "node:path";
import { mkdtempSync } from "node:fs";
import os from "node:os";

test("Electron launches and connects to Python", async () => {
  const root = path.resolve(__dirname, "..", "..");
  const userData = mkdtempSync(path.join(os.tmpdir(), "neural-debugger-"));
  const electronApp = await electron.launch({
    args: [path.join(root, "dist-electron", "main.js")],
    env: {
      ...process.env,
      NEURAL_DEBUGGER_USER_DATA: userData,
      NEURAL_DEBUGGER_PYTHON: process.env.PYTHON ?? "python"
    }
  });

  try {
    const page = await electronApp.firstWindow();
    await expect(page.getByRole("heading", { name: "Neural Debugger" })).toBeVisible();
    await expect(page.getByText("Python connected", { exact: true })).toBeVisible({
      timeout: 15_000
    });

    const settings = await page.evaluate(() =>
      window.desktopApi.updateSettings({
        theme: "dark",
        gpuEnabled: true,
        cachePath: "C:/cache",
        modelPath: "gpt2",
        workspacePath: "C:/workspace"
      })
    );
    expect(settings.theme).toBe("dark");
    expect(settings.gpuEnabled).toBe(true);
  } finally {
    await electronApp.close();
  }
});
