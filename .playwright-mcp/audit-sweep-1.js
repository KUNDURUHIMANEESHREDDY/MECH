async (page) => {
  const report = { steps: [], errors: [], failedRequests: [], views: {} };
  const t0 = Date.now();

  page.on('console', (msg) => {
    if (msg.type() === 'error') report.errors.push(`[console] ${msg.text()}`);
  });
  page.on('pageerror', (err) => report.errors.push(`[pageerror] ${err.message}`));
  page.on('requestfailed', (req) =>
    report.failedRequests.push(`[reqfail] ${req.url()} :: ${req.failure()?.errorText || 'unknown'}`));
  page.on('response', (res) => {
    if (res.status() >= 400) report.failedRequests.push(`[http ${res.status()}] ${res.url()}`);
  });
  page.on('dialog', async (d) => {
    report.steps.push(`DIALOG: ${d.type()} "${d.message()}" -> ACCEPTED`);
    await d.accept();
  });

  const wait = (ms) => page.waitForTimeout(ms);

  // 1. Load a model
  const loadBtn = page.getByRole('button', { name: 'Load gpt2-small' }).first();
  await loadBtn.click().catch((e) => report.steps.push(`CLICK FAIL load-btn: ${e.message}`));
  await wait(6000);
  report.steps.push(`after model load, body contains "Choose a model": ${await page.getByText('Choose a model to load').count()}`);

  // 2. The 27 sidebar views
  const views = [
    ['Model Explorer', 'explorer'], ['GPT-2 Live', 'gpt2'], ['Transformer Visualizer', 'transformer'],
    ['Workspace', 'workspace'], ['Neural Explorer', 'neuralexplorer'], ['Knowledge Graph', 'knowledgegraph'],
    ['Circuit Explorer', 'circuitexplorer'], ['Models', 'models'], ['Prompts', 'prompts'],
    ['Debugger', 'debugger'], ['Build Log', 'build'], ['Benchmark', 'benchmark'], ['Benchmark Suite', 'benchmarksuite'],
    ['Experiments', 'experiments'], ['Sessions', 'sessions'], ['Reports', 'reports'],
    ['Reasoning Trace', 'reasoning'], ['Evidence Fusion', 'evidencefusion'], ['Analytics', 'analytics'],
    ['Health', 'health'], ['Campaigns', 'campaigns'], ['Plugins', 'plugins'],
    ['Research Notebook', 'notebook'], ['Lab Notebook', 'labnotebook'], ['Paper Reproduction', 'reproduction'],
    ['Settings', 'settings'], ['Logging', 'logging'], ['Projects', 'projects'], ['Recent Files', 'recent'],
  ];

  for (const [label, key] of views) {
    const vStart = Date.now();
    const viewRec = { buttons: [], clicked: 0, skipped: 0, errors: [] };
    const btn = page.getByRole('button', { name: label, exact: true }).last();
    await btn.click().catch((e) => viewRec.errors.push(`nav click fail: ${e.message}`));
    await wait(2500);

    // Record the main content area buttons
    const buttons = page.locator('button').all();
    const all = await buttons;
    const seen = new Set();
    for (const b of all) {
      if (!(await b.isVisible().catch(() => false))) continue;
      const txt = ((await b.textContent().catch(() => '')) || '').trim();
      const title = (await b.getAttribute('title').catch(() => '')) || '';
      const name = txt || title || 'unnamed';
      if (seen.has(name) || name.length === 0) continue;
      seen.add(name);
      viewRec.buttons.push(name);
    }

    // Click each button (skip nav/sidebar/topbar chrome)
    const skipRe = /^(Explorer|Search|Debug|Code|Toggle sidebar|Collapse activity bar|Collapse sidebar|M)$/i;
    for (const name of viewRec.buttons) {
      if (skipRe.test(name)) { viewRec.skipped++; continue; }
      // re-locate each time
      const locator = page.getByRole('button', { name, exact: false }).first();
      try {
        await locator.scrollIntoViewIfNeeded();
        await locator.click();
        viewRec.clicked++;
        await wait(1200);
      } catch (e) {
        viewRec.errors.push(`click "${name}": ${e.message.split('\n')[0]}`);
      }
    }

    // view-level console snapshot
    viewRec.errors.push(...report.errors.slice(-0));
    report.views[key] = { label, ms: Date.now() - vStart, ...viewRec };
    report.steps.push(`VISITED ${label} in ${Date.now() - vStart}ms`);
    // screenshot evidence
    await page.screenshot({ path: `.playwright-mcp/audit-${key}.png`, type: 'png' }).catch(() => {});
  }

  report.totalMs = Date.now() - t0;
  return JSON.stringify(report);
}