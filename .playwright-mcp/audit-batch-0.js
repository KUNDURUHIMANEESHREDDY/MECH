// Batch audit runner. Usage: set BATCH (0-based) then run via browser_run_code_unsafe.
async (page) => {
  const BATCH = Number(0);
  const BATCHES = [
    [['Model Explorer', 'explorer'], ['GPT-2 Live', 'gpt2'], ['Transformer Visualizer', 'transformer'], ['Workspace', 'workspace'], ['Neural Explorer', 'neuralexplorer']],
    [['Knowledge Graph', 'knowledgegraph'], ['Circuit Explorer', 'circuitexplorer'], ['Models', 'models'], ['Prompts', 'prompts'], ['Debugger', 'debugger']],
    [['Build Log', 'build'], ['Benchmark', 'benchmark'], ['Benchmark Suite', 'benchmarksuite'], ['Experiments', 'experiments'], ['Sessions', 'sessions']],
    [['Reports', 'reports'], ['Reasoning Trace', 'reasoning'], ['Evidence Fusion', 'evidencefusion'], ['Analytics', 'analytics'], ['Health', 'health']],
    [['Campaigns', 'campaigns'], ['Plugins', 'plugins'], ['Research Notebook', 'notebook'], ['Lab Notebook', 'labnotebook'], ['Paper Reproduction', 'reproduction']],
    [['Settings', 'settings'], ['Logging', 'logging'], ['Projects', 'projects'], ['Recent Files', 'recent']],
  ];
  const views = BATCHES[BATCH] ?? [];

  const report = { batch: BATCH, views: {}, errors: [], failed: [] };
  const consoleErrors = [];
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push(`PAGEERROR: ${e.message}`));
  page.on('requestfailed', (r) => report.failed.push(`[reqfail] ${r.url()} :: ${r.failure()?.errorText || ''}`));
  page.on('response', (r) => { if (r.status() >= 400) report.failed.push(`[http ${r.status()}] ${r.url()}`); });
  const wait = (ms) => page.waitForTimeout(ms);
  const skipRe = /^(Explorer|Search|Debug|Code|Toggle sidebar|Collapse activity bar|Collapse sidebar|M)$/i;

  for (const [label, key] of views) {
    const st = Date.now();
    const rec = { buttons: [], clicked: [], skipped: [], clickFails: [], navFail: null };
    const nav = page.getByRole('button', { name: key === 'gpt2' ? 'GPT-2 Live' : label, exact: true });
    try { await nav.click(); } catch (e) { rec.navFail = e.message.split('\n')[0]; }
    await wait(2200);

    const beforeErr = consoleErrors.length;
    const btns = await page.locator('button').all();
    const seen = new Set();
    for (const b of btns) {
      if (!(await b.isVisible().catch(() => false))) continue;
      const txt = ((await b.textContent().catch(() => '')) || '').trim();
      const title = (await b.getAttribute('title').catch(() => '')) || '';
      const name = txt || title || '';
      if (!name || seen.has(name)) continue;
      seen.add(name);
      rec.buttons.push(name);
    }
    for (const name of rec.buttons) {
      if (skipRe.test(name)) { rec.skipped.push(name); continue; }
      const loc = page.getByRole('button', { name, exact: false }).first();
      try {
        await loc.scrollIntoViewIfNeeded();
        await loc.click();
        rec.clicked.push(name);
        await wait(800);
      } catch (e) {
        rec.clickFails.push(`${name} :: ${e.message.split('\n')[0]}`);
      }
    }
    rec.newErrors = consoleErrors.slice(beforeErr);
    rec.ms = Date.now() - st;
    report.views[key] = rec;
    try { await page.screenshot({ path: `.playwright-mcp/audit-${key}.png`, type: 'png' }); } catch {}
  }
  report.errors = consoleErrors;
  return JSON.stringify(report);
}