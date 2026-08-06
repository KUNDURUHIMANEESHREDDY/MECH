/**
 * Focused diagnostic: capture detailed Vue/Console/Network errors
 * when navigating between pages.
 */
const { chromium } = require('@playwright/test');
const fs = require('fs');

const PAGES = [
  'explorer', 'gpt2', 'gpt2explorer', 'transformer', 'transformerExplorer',
  'workspace', 'models', 'prompts', 'debugger', 'experiments', 'sessions',
  'reports', 'settings', 'logging', 'build', 'neuralexplorer', 'benchmark',
  'benchmarksuite', 'knowledgegraph', 'circuitexplorer', 'reasoning',
  'evidencefusion', 'campaigns', 'analytics', 'health', 'plugins',
  'notebook', 'labnotebook', 'reproduction', 'projects', 'recent',
];

(async () => {
  const browser = await chromium.launch({ headless: true, slowMo: 50 });
  const context = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await context.newPage();

  const allErrors = [];
  const consoleMsgs = [];
  const networkErrs = [];

  page.on('console', msg => {
    if (msg.type() === 'error') {
      const details = msg.text();
      // Try to get stack trace
      const loc = msg.location();
      const entry = `Console: ${details} @ ${loc.url}:${loc.lineNumber}:${loc.columnNumber}`;
      consoleMsgs.push(entry);
      allErrors.push(entry);
    }
  });

  page.on('pageerror', err => {
    const entry = `PageError: ${err.message}\n  Stack: ${err.stack}`;
    allErrors.push(entry);
    console.log(`  [PAGE ERROR] ${err.message}`);
    if (err.stack) console.log(`    ${err.stack.split('\n').slice(1, 4).join('\n    ')}`);
  });

  page.on('requestfailed', req => {
    const fail = req.failure();
    if (fail) {
      const entry = `Network: ${req.method()} ${req.url()} -> ${fail.errorText}`;
      networkErrs.push(entry);
      allErrors.push(entry);
    }
  });

  console.log('Navigating to app root...');
  await page.goto('http://localhost:5173/', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(3000);

  // Load model
  const loadBtn = page.locator('button:has-text("Load")');
  if (await loadBtn.count() > 0) {
    console.log('Loading model...');
    await loadBtn.first().click();
    await page.waitForTimeout(5000);
  }

  // Navigate through pages and capture errors
  for (const pageName of PAGES) {
    console.log(`\n--- ${pageName} ---`);
    const beforeErrors = allErrors.length;

    await page.evaluate(p => { window.location.hash = p; }, pageName);
    await page.waitForTimeout(3000);

    const newErrors = allErrors.slice(beforeErrors);
    if (newErrors.length > 0) {
      console.log(`  ${newErrors.length} new errors:`);
      newErrors.forEach(e => console.log(`    ${e}`));
    } else {
      console.log('  No errors');
    }

    // Check for "Coming soon" fallback
    const comingSoon = page.locator('text="Coming soon"');
    const hintText = page.locator('.welcome-hint');
    if (await comingSoon.count() > 0) {
      console.log('  Status: Coming soon fallback');
    } else if (await hintText.count() > 0) {
      console.log('  Status: Welcome hint visible');
    } else {
      console.log('  Status: Component rendered');
    }

    // Count visible buttons
    const btnCount = await page.locator('button:visible').count();
    console.log(`  Visible buttons: ${btnCount}`);

    // Take a screenshot for the first few pages
    if (['workspace', 'models', 'settings'].includes(pageName)) {
      await page.screenshot({ path: `screenshots/${pageName}.png` });
    }
  }

  console.log('\n\n========== ERROR SUMMARY ==========');
  console.log(`Console errors: ${consoleMsgs.length}`);
  console.log(`Page errors: ${allErrors.filter(e => e.startsWith('PageError')).length}`);
  console.log(`Network errors: ${networkErrs.length}`);

  // Write full report
  let report = '========== FULL ERROR REPORT ==========\n\n';
  report += `Console errors (${consoleMsgs.length}):\n`;
  consoleMsgs.forEach(e => report += `  ${e}\n`);
  report += `\nPage errors (${allErrors.filter(e => e.startsWith('PageError')).length}):\n`;
  allErrors.filter(e => e.startsWith('PageError')).forEach(e => report += `  ${e}\n`);
  report += `\nNetwork errors (${networkErrs.length}):\n`;
  networkErrs.forEach(e => report += `  ${e}\n`);

  if (!fs.existsSync('screenshots')) fs.mkdirSync('screenshots');
  fs.writeFileSync('playwright_error_report.txt', report);
  console.log('\nFull report written to playwright_error_report.txt');

  await browser.close();
})().catch(err => {
  console.error('FATAL:', err);
  process.exit(1);
});
