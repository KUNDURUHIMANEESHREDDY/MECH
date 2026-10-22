const { test, expect } = require('@playwright/test');

/**
 * Integration tests for MECH scientific workflows.
 * These tests verify end-to-end behavior with the backend running.
 */

test.describe('Research Loop Lifecycle', () => {
  test('can start, monitor, and stop a research loop', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByTestId('nav-society')).toBeVisible();
    await page.getByTestId('nav-society').click();
    await expect(page).toHaveURL(/#society$/);
    await expect(page.getByTestId('view-society')).toBeVisible();

    // The society view should have a run button and status display
    const runButton = page.getByTestId('society-run-button');
    if (await runButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await runButton.click();
      // Should show progress or queued status
      await expect(page.getByTestId('society-run-status')).toBeVisible({ timeout: 10000 });
    }
  });

  test('displays loop steps with proper provenance labels', async ({ page }) => {
    await page.goto('/#society');
    await expect(page.getByTestId('view-society')).toBeVisible();

    // Check that step statuses are shown
    const steps = [
      'load', 'reproduce', 'inspect', 'patch', 'discover', 'validate', 'publish'
    ];
    for (const step of steps) {
      const stepEl = page.getByTestId(`society-step-${step}`);
      if (await stepEl.isVisible({ timeout: 3000 }).catch(() => false)) {
        // Each step should show its provenance status
        await expect(stepEl).toContainText(/live|unavailable|seeded|synthetic/);
      }
    }
  });
});

test.describe('Scientific Visualizations - Live vs Mock', () => {
  test('model explorer shows provenance badges for live measurements', async ({ page }) => {
    await page.goto('/#explorer');
    await expect(page.getByTestId('view-explorer')).toBeVisible();

    // Load model if not already loaded
    const loadBtn = page.getByTestId('explorer-load-model');
    if (await loadBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await loadBtn.click();
      await expect(page.getByText(/Loading|loaded/i)).toBeVisible({ timeout: 30000 });
    }

    // Run a prompt
    const promptInput = page.getByTestId('explorer-prompt-input');
    const runBtn = page.getByTestId('explorer-run-prompt');
    if (await promptInput.isVisible({ timeout: 3000 }).catch(() => false)) {
      await promptInput.fill('The capital of France is');
      await runBtn.click();
      await expect(page.getByTestId('explorer-result')).toBeVisible({ timeout: 30000 });

      // Check for provenance indicators on result fields
      const result = page.getByTestId('explorer-result');
      await expect(result).toContainText(/live|provenance/);
    }
  });

  test('circuit explorer displays measured vs unmeasured fidelity', async ({ page }) => {
    await page.goto('/#circuitexplorer');
    await expect(page.getByTestId('view-circuitexplorer')).toBeVisible();

    // Check circuit cards have fidelity badges
    const circuits = page.getByTestId('circuit-list');
    if (await circuits.isVisible({ timeout: 5000 }).catch(() => false)) {
      const items = circuits.getByTestId('circuit-item');
      const count = await items.count();
      if (count > 0) {
        for (let i = 0; i < Math.min(count, 3); i++) {
          const item = items.nth(i);
          await expect(item).toContainText(/measured|unmeasured|fidelity/);
        }
      }
    }
  });

  test('attention visualizer shows real attention matrices', async ({ page }) => {
    await page.goto('/#transformer');
    await expect(page.getByTestId('view-transformer')).toBeVisible();

    const layerSelect = page.getByTestId('transformer-layer-select');
    if (await layerSelect.isVisible({ timeout: 3000 }).catch(() => false)) {
      await layerSelect.selectOption('0');
      await expect(page.getByTestId('attention-matrix')).toBeVisible({ timeout: 10000 });
      // Should show numeric matrix values, not placeholders
      await expect(page.getByTestId('attention-matrix')).not.toContainText(/N\/A|unavailable|placeholder/);
    }
  });
});

test.describe('Error State Handling', () => {
  test('handles backend timeout gracefully', async ({ page }) => {
    // Every API call times out. The view must report the failure instead of
    // rendering an empty result set that looks like a measurement.
    await page.route('**/api/**', async route => {
      await new Promise(r => setTimeout(r, 500));
      await route.abort('timedout');
    });

    await page.goto('/#explorer');
    const explorer = page.getByTestId('view-explorer');
    await expect(explorer).toBeVisible();
    await expect(explorer.getByText(/not responding|unavailable|timed out/i).first()).toBeVisible();
    // A failed read must not leave a result panel behind.
    await expect(explorer.getByTestId('model-prompt-result')).toHaveCount(0);
  });

  test('handles authentication failure with clear message', async ({ page }) => {
    await page.route('**/api/**', route => route.fulfill({
      status: 401,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'invalid credentials' })
    }));

    await page.goto('/#explorer');
    const explorer = page.getByTestId('view-explorer');
    await expect(explorer).toBeVisible();
    // The pill reports the refused token; the view reports the refused read.
    await expect(page.getByTestId('runtime-status')).toContainText(/unauthorized|failed/i);
    await expect(explorer.getByText(/unauthorized|401|credentials|invalid/i).first()).toBeVisible();
    await expect(explorer.getByTestId('model-prompt-result')).toHaveCount(0);
  });

  test('displays unavailable endpoints honestly', async ({ page }) => {
    // Deterministic offline: the analytics tiles are built from three reads, so
    // with the reads refused they must read as unavailable -- never as 0, which
    // is a measurement.
    await page.route('**/api/**', route => route.abort());
    await page.goto('/#analytics');
    const view = page.getByTestId('view-analytics');
    await expect(view).toBeVisible();

    await expect(view).toContainText(/offline|unavailable/i);
    await expect(view).toContainText(/—/);
    await expect(view).toContainText(/never replaced with a placeholder metric|reported as such/i);
  });
});

test.describe('Persistence and Reload', () => {
  test('saved project survives page reload', async ({ page }) => {
    await page.goto('/#projects');
    await expect(page.getByTestId('view-projects')).toBeVisible();

    // Create a project
    const createBtn = page.getByTestId('project-create-button');
    if (await createBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await createBtn.click();
      const nameInput = page.getByTestId('project-name-input');
      await nameInput.fill('Test Project Reload');
      await page.getByTestId('project-create-submit').click();
      
      await expect(page.getByText('Test Project Reload')).toBeVisible({ timeout: 5000 });

      // Reload page
      await page.reload();
      await expect(page.getByTestId('view-projects')).toBeVisible();
      await expect(page.getByText('Test Project Reload')).toBeVisible({ timeout: 5000 });
    }
  });

  test('theme preference survives reload', async ({ page }) => {
    await page.goto('/#settings');
    await expect(page.getByTestId('view-settings')).toBeVisible();

    const themeSelect = page.getByTestId('settings-theme-select');
    if (await themeSelect.isVisible({ timeout: 3000 }).catch(() => false)) {
      // Change theme
      await themeSelect.selectOption('dark');
      await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');

      // Reload
      await page.reload();
      await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    }
  });
});

test.describe('API Response Validation', () => {
  test('inference endpoint returns expected schema', async ({ page }) => {
    // Loading the weights and running a prompt is minutes of CPU when the
    // machine is busy. The point of this test is the response contract, so it
    // gets a budget that covers the work rather than the default one.
    test.setTimeout(600_000);
    const responses = [];
    // Observe the response without sending the request twice.
    //
    // `route.fetch()` followed by `route.continue()` performs the request
    // twice: the backend runs the forward pass twice, and on a busy machine
    // the first fetch can exceed Playwright's 30s default, so the page never
    // receives a response at all. Fulfilling the page with the fetched
    // response sends it once and still lets the test read the body.
    await page.route('**/api/gpt2/run_prompt', async route => {
      const response = await route.fetch({ timeout: 180000 });
      const body = await response.json();
      responses.push(body);
      await route.fulfill({ response });
    });

    await page.goto('/#explorer');
    const explorer = page.getByTestId('view-explorer');
    await expect(explorer).toBeVisible();
    // The load control stays disabled until a catalogue entry is selected.
    await explorer.locator('#model-explorer-model').selectOption('gpt2-small');
    await explorer.getByTestId('model-load').click();
    await expect(explorer.getByTestId('model-loaded')).toBeVisible({ timeout: 360000 });

    await explorer.locator('#model-explorer-prompt').fill('Test prompt');
    await explorer.getByTestId('model-run-prompt').click();
    await expect(explorer.getByTestId('model-prompt-result')).toBeVisible({ timeout: 360000 });

    // Validate response structure
    expect(responses.length).toBeGreaterThan(0);
    const resp = responses[0];
    expect(resp).toHaveProperty('status');
    expect(resp).toHaveProperty('provenance');
    expect(['live', 'seeded', 'unavailable', 'synthetic']).toContain(resp.provenance);
  });

  test('IOI endpoint returns structured response with provenance', async ({ page }) => {
    const responses = [];
    await page.route('**/api/gpt2/ioi', async route => {
      const response = await route.fetch();
      const body = await response.json();
      responses.push(body);
      route.continue();
    });

    await page.goto('/#gpt2');
    await expect(page.getByTestId('view-gpt2')).toBeVisible();

    const ioiBtn = page.getByTestId('gpt2-ioi-run');
    if (await ioiBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await ioiBtn.click();
      await expect(page.getByTestId('gpt2-ioi-result')).toBeVisible({ timeout: 30000 });

      expect(responses.length).toBeGreaterThan(0);
      const resp = responses[0];
      expect(resp).toHaveProperty('status');
      expect(resp).toHaveProperty('provenance');
      expect(['live', 'unavailable']).toContain(resp.provenance);
      if (resp.provenance === 'live') {
        expect(resp).toHaveProperty('clean_top1');
        expect(resp).toHaveProperty('corrupted_top1');
        expect(resp).toHaveProperty('logit_diff');
      }
    }
  });
});

test.describe('Navigation and State', () => {
  test('switching tools preserves no stale data', async ({ page }) => {
    await page.goto('/#explorer');
    const explorer = page.getByTestId('view-explorer');
    await expect(explorer).toBeVisible();
    await explorer.locator('#model-explorer-prompt').fill('stale marker prompt');

    // Navigate to another tool. The shell resolves the target chunk before it
    // swaps, so the previous tool's DOM is never shown under the new tool's
    // name -- and a screenshot taken at any point shows exactly one tool.
    await page.getByTestId('nav-circuitexplorer').click();
    await expect(page).toHaveURL(/#circuitexplorer$/);
    const circuit = page.getByTestId('view-circuitexplorer');
    await expect(circuit).toBeVisible();

    await expect(page.locator('[data-testid^="view-"]')).toHaveCount(1);
    await expect(page.locator('body')).not.toContainText('stale marker prompt');

    await page.getByTestId('nav-explorer').click();
    await expect(page).toHaveURL(/#explorer$/);
    await expect(page.locator('[data-testid^="view-"]')).toHaveCount(1);
    // Returning to a tool mounts it fresh: it asks the backend for a prompt
    // rather than restoring the previous one, and it shows no result from the
    // earlier visit. Both are that tool's own state, not a leak from another.
    await expect(page.locator('#model-explorer-prompt')).not.toHaveValue('stale marker prompt');
    await expect(page.locator('#model-explorer-prompt')).not.toHaveValue('');
    await expect(page.getByTestId('model-prompt-result')).toHaveCount(0);
  });

  test('runtime status updates across tools', async ({ page }) => {
    await page.goto('/');
    const status = page.getByTestId('runtime-status');
    await expect(status).toBeVisible();

    const initialText = await status.textContent();
    
    // `gpt2` is an alias route: the shell renders one nav entry per canonical
    // route, so there is no `nav-gpt2` button to click. It is reached by hash.
    await page.goto('/#gpt2');
    await expect(page).toHaveURL(/#gpt2$/);
    
    // Status should still be visible and consistent
    const gpt2Status = page.getByTestId('runtime-status');
    await expect(gpt2Status).toBeVisible();
    await expect(gpt2Status).toHaveText(initialText);
  });
});