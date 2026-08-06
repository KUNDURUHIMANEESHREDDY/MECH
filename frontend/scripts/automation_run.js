/**
 * Full automation: navigate every page, click every button, fill every input,
 * and collect all console/network errors.
 */
const { chromium } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const PAGES = [
  'explorer', 'gpt2', 'gpt2explorer', 'transformer', 'transformerExplorer',
  'workspace', 'models', 'prompts', 'debugger', 'experiments', 'sessions',
  'reports', 'settings', 'logging', 'build', 'neuralexplorer', 'benchmark',
  'benchmarksuite', 'knowledgegraph', 'circuitexplorer', 'reasoning',
  'evidencefusion', 'campaigns', 'analytics', 'health', 'plugins',
  'notebook', 'labnotebook', 'reproduction', 'projects', 'recent',
];

const logFile = path.join(__dirname, 'automation_results.log');
let logBuf = '';

function log(msg) {
  console.log(msg);
  logBuf += msg + '\n';
}

(async () => {
  const browser = await chromium.launch({ headless: false, slowMo: 100 });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
  });

  const consoleErrors = [];
  const pageErrors = [];
  const networkErrors = [];

  const page = await context.newPage();

  page.on('console', msg => {
    if (msg.type() === 'error') {
      const text = msg.text();
      consoleErrors.push(text);
      log(`  [CONSOLE ERROR] ${text}`);
    }
  });

  page.on('pageerror', err => {
    pageErrors.push(err.message);
    log(`  [PAGE ERROR] ${err.message}`);
  });

  page.on('requestfailed', req => {
    const err = req.failure();
    if (err) {
      networkErrors.push(`${req.method()} ${req.url()} -> ${err.errorText}`);
      log(`  [NETWORK ERROR] ${req.method()} ${req.url()} -> ${err.errorText}`);
    }
  });

  log('Navigating to app root...');
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle', timeout: 30000 });

  // Wait for the app to mount
  await page.waitForTimeout(2000);

  // Check if ErrorUI overlay is present and dismiss it
  const dismissBtn = page.locator('button.error-dismiss-btn');
  if (await dismissBtn.count() > 0) {
    log('  Dismissing ErrorUI overlay...');
    await dismissBtn.first().click();
    await page.waitForTimeout(500);
  }

  // For the explorer page: try loading a model then running inference
  log('\n--- Testing Model Explorer (explorer page) ---');
  const modelButtons = page.locator('button');
  const btnCount = await modelButtons.count();
  log(`  Found ${btnCount} buttons on initial page`);

  // Try clicking model load buttons
  for (let i = 0; i < btnCount; i++) {
    try {
      const btn = modelButtons.nth(i);
      const text = await btn.textContent();
      const isVisible = await btn.isVisible();
      if (isVisible && text?.includes('Load')) {
        log(`  Clicking: ${text.trim()}`);
        await btn.click();
        await page.waitForTimeout(3000);
        // Check for errors
        break; // Just load one model
      }
    } catch (e) {
      log(`  Error clicking button ${i}: ${e.message}`);
    }
  }

  // After loading model, try the Run button
  const runBtn = page.locator('button:has-text("Run")');
  if (await runBtn.count() > 0) {
    log('  Clicking Run button...');
    await runBtn.first().click();
    await page.waitForTimeout(5000);
  }

  // Now navigate through all pages
  for (const pageName of PAGES) {
    log(`\n--- Navigating to page: ${pageName} ---`);

    // Navigate via hash
    await page.evaluate(p => { window.location.hash = p; }, pageName);
    await page.waitForTimeout(3000);

    // Dismiss any error overlay
    const errDismiss = page.locator('button.error-dismiss-btn');
    if (await errDismiss.count() > 0) {
      log(`  Dismissing error on ${pageName}...`);
      await errDismiss.first().click();
      await page.waitForTimeout(500);
    }

    // Click every visible button on the page
    const buttons = page.locator('button');
    const count = await buttons.count();
    log(`  Found ${count} buttons on ${pageName}`);

    for (let i = 0; i < count; i++) {
      try {
        const btn = buttons.nth(i);
        if (!await btn.isVisible()) continue;
        const text = (await btn.textContent()).trim();
        if (!text) continue;

        // Skip dismiss buttons, toggles we already handle
        const styles = await btn.getAttribute('class') || '';
        log(`  - Clicking button: "${text}" (${i+1}/${count})`);
        await btn.scrollIntoViewIfNeeded();
        await btn.click({ timeout: 5000 });
        await page.waitForTimeout(1000);
      } catch (e) {
        log(`  ERROR clicking button "${text}" on ${pageName}: ${e.message}`);
      }
    }

    // Fill any input/textarea
    const inputs = page.locator('input, textarea');
    const inputCount = await inputs.count();
    for (let i = 0; i < inputCount; i++) {
      try {
        const input = inputs.nth(i);
        if (!await input.isVisible()) continue;
        await input.scrollIntoViewIfNeeded();
        await input.fill(`Test input ${i}`);
        await page.waitForTimeout(300);
      } catch (e) {
        log(`  Error filling input on ${pageName}: ${e.message}`);
      }
    }
  }

  // Summary
  log('\n' + '='.repeat(60));
  log('AUTOMATION SUMMARY');
  log('='.repeat(60));
  log(`Console errors: ${consoleErrors.length}`);
  log(`Page errors: ${pageErrors.length}`);
  log(`Network errors: ${networkErrors.length}`);

  if (consoleErrors.length > 0) {
    log('\nConsole Errors:');
    consoleErrors.forEach(e => log(`  - ${e}`));
  }
  if (pageErrors.length > 0) {
    log('\nPage Errors:');
    pageErrors.forEach(e => log(`  - ${e}`));
  }
  if (networkErrors.length > 0) {
    log('\nNetwork Errors:');
    networkErrors.forEach(e => log(`  - ${e}`));
  }

  fs.writeFileSync(logFile, logBuf);
  log(`\nFull results written to ${logFile}`);

  await browser.close();
  process.exit(0);
})().catch(err => {
  console.error('FATAL:', err);
  fs.appendFileSync(logFile, `\nFATAL: ${err.stack}\n`);
  process.exit(1);
});
