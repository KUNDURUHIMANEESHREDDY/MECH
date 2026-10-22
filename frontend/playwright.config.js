const { defineConfig } = require('@playwright/test');

const port = Number(process.env.MECH_PLAYWRIGHT_PORT || 5173);
const baseURL = `http://localhost:${port}`;

module.exports = defineConfig({
  testDir: './tests/playwright',
  // The suite runs against a Vite dev server, which compiles a route's chunk on
  // first request. A cold route could take longer than the old 30s budget while
  // seven workers hit the compiler at once, which read as a product failure.
  // Per test. The specs that do real work set their own budget; this is the
  // default for the rest.
  timeout: 120_000,
  // Per-assertion budget: a freshly compiled route chunk can take several
  // seconds to mount under parallel load.
  // Per assertion: long enough for a cold route chunk to compile on a loaded
  // machine, short enough to fail fast on a real defect.
  expect: { timeout: 20_000 },
  fullyParallel: false,
  reporter: [['list']],
  use: {
    baseURL,
    trace: 'on-first-retry'
  },
  webServer: {
    command: `npm run dev:renderer -- --port ${port}`,
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000
  }
});
