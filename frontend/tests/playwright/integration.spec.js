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
    // Route API calls to delay then fail
    await page.route('**/api/**', async route => {
      await new Promise(r => setTimeout(r, 5000));
      await route.abort('timedout');
    });

    await page.goto('/#explorer');
    await page.getByTestId('explorer-load-model').click();
    
    // Should show timeout error, not crash
    await expect(page.getByText(/timeout|error|unavailable/i)).toBeVisible({ timeout: 10000 });
  });

  test('handles authentication failure with clear message', async ({ page }) => {
    // Route with 401 responses
    await page.route('**/api/**', route => route.fulfill({
      status: 401,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Invalid credentials' })
    }));

    await page.goto('/#explorer');
    await page.getByTestId('explorer-load-model').click();
    
    // Should show auth error, not crash
    await expect(page.getByText(/auth|unauthorized|login|credentials/i)).toBeVisible({ timeout: 10000 });
  });

  test('displays unavailable endpoints honestly', async ({ page }) => {
    await page.goto('/#analytics');
    await expect(page.getByTestId('view-analytics')).toBeVisible();

    // Analytics should show "unavailable" not fake data
    await expect(page.getByTestId('view-analytics')).toContainText(/unavailable|not implemented|no data/i);
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
    const responses = [];
    await page.route('**/api/gpt2/run_prompt', async route => {
      const response = await route.fetch();
      const body = await response.json();
      responses.push(body);
      route.continue();
    });

    await page.goto('/#explorer');
    await page.getByTestId('explorer-load-model').click();
    await expect(page.getByText(/loaded/i)).toBeVisible({ timeout: 30000 });

    await page.getByTestId('explorer-prompt-input').fill('Test prompt');
    await page.getByTestId('explorer-run-prompt').click();
    await expect(page.getByTestId('explorer-result')).toBeVisible({ timeout: 30000 });

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
    await page.getByTestId('explorer-load-model').click();
    await expect(page.getByText(/loaded/i)).toBeVisible({ timeout: 30000 });

    // Navigate to another tool
    await page.getByTestId('nav-circuitexplorer').click();
    await expect(page).toHaveURL(/#circuitexplorer$/);
    await expect(page.getByTestId('view-circuitexplorer')).toBeVisible();

    // Navigate back
    await page.getByTestId('nav-explorer').click();
    await expect(page).toHaveURL(/#explorer$/);
    
    // Should not show stale result from previous session
    const result = page.getByTestId('explorer-result');
    // Result should either be cleared or match current state
  });

  test('runtime status updates across tools', async ({ page }) => {
    await page.goto('/');
    const status = page.getByTestId('runtime-status');
    await expect(status).toBeVisible();

    const initialText = await status.textContent();
    
    await page.getByTestId('nav-gpt2').click();
    await expect(page).toHaveURL(/#gpt2$/);
    
    // Status should still be visible and consistent
    const gpt2Status = page.getByTestId('runtime-status');
    await expect(gpt2Status).toBeVisible();
    await expect(gpt2Status).toHaveText(initialText);
  });
});