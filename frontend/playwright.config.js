const { defineConfig } = require('@playwright/test');

const port = Number(process.env.MECH_PLAYWRIGHT_PORT || 5173);
const baseURL = `http://localhost:${port}`;

module.exports = defineConfig({
  testDir: './tests/playwright',
  timeout: 30_000,
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
