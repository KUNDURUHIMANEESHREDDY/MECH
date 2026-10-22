const { defineConfig } = require('@playwright/test');

const port = Number(process.env.MECH_PLAYWRIGHT_PORT || 5173);
const baseURL = `http://localhost:${port}`;

module.exports = defineConfig({
  testDir: './tests/playwright',
  // Per test. The specs that do real work -- 36 routes, a model load, a real
  // Society run -- carry their own budget; this is the default for the rest.
  timeout: 120_000,
  // Per assertion: long enough for a cold route chunk to compile on a loaded
  // machine, short enough to fail fast on a real defect.
  expect: { timeout: 20_000 },
  fullyParallel: false,
  // The suite drives ONE backend, and that backend's inference is CPU-bound:
  // a Society run sweeps every attention head of GPT-2 on every prompt. With
  // more browsers than the machine can serve, the slow specs were the ones
  // that failed -- the product looked broken when it was merely contended.
  // Four workers is the honest setting for a suite that shares one server.
  workers: Number(process.env.MECH_PLAYWRIGHT_WORKERS || 4),
  reporter: [['list']],
  // The weights are loaded once, before any spec starts, so that the specs
  // which need GPT-2 do not each queue behind the same cold load.
  globalSetup: require.resolve('./tests/playwright/global-setup.js'),
  use: {
    baseURL,
    trace: 'on-first-retry'
  },
  projects: [
    {
      name: 'desktop',
      testIgnore: /society-lifecycle\.spec\.js/,
    },
    {
      // The Society lifecycle runs on its own, after the rest of the suite:
      // it is a real CPU-bound sweep, and running it beside the other specs
      // made both slower than their budgets.
      name: 'society',
      testMatch: /society-lifecycle\.spec\.js/,
      dependencies: ['desktop'],
    },
  ],
  webServer: {
    command: `npm run dev:renderer -- --port ${port}`,
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000}
});
