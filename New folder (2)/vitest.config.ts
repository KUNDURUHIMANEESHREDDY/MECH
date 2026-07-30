import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "electron/renderer/test/setup.ts",
    include: ["tests/frontend/**/*.test.ts?(x)"]
  }
});
