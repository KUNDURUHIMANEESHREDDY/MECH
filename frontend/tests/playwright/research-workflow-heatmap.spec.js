const { test, expect } = require('@playwright/test');

const fingerprint = 'sha256:' + 'a'.repeat(64);
function capture({ prompt = 'A', fingerprintValue = fingerprint, attested = true, provenance = 'live', matrix = [[1, 2], [3, 4]], status = 'ok' } = {}) {
  return {
    status,
    provenance,
    attested,
    resid_post: matrix,
    tokens: [prompt + '-token-0', prompt + '-token-1'],
    measurement_record: {
      model_identity: {
        name: 'gpt2',
        content_hash: fingerprintValue,
        content_hash_source: 'explicit',
      },
    },
  };
}

async function openHeatmap(page, responses) {
  let call = 0;
  await page.route('**/api/gpt2/layer_activations', async route => {
    const response = responses[call++];
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(response) });
  });
  await page.goto('/#researchworkflow');
  const panel = page.getByTestId('residual-activation-comparison');
  await expect(panel).toBeVisible();
  await panel.getByRole('button', { name: 'Capture residual difference' }).click();
  return { panel, calls: () => call };
}

test('Research Workflow captures both prompts and renders finite residual deltas with matching live fingerprints', async ({ page }) => {
  const { panel, calls } = await openHeatmap(page, [
    capture({ prompt: 'A', matrix: [[1, 2], [3, 4]] }),
    capture({ prompt: 'B', matrix: [[1.5, 1], [5, 2]] }),
  ]);

  await expect(panel.getByText(/weight fingerprints matched/i)).toBeVisible();
  await expect(panel.getByText(/gpt2/).first()).toBeVisible();
  await expect(panel.getByRole('cell', { name: '0.5' })).toBeVisible();
  await expect(panel.getByRole('cell', { name: '-1' })).toBeVisible();
  await expect(panel).toContainText('A-token-0');
  await expect(panel).toContainText('B-token-0');
  expect(calls()).toBe(2);
});

test('Research Workflow refuses missing or fallback fingerprint evidence', async ({ page }) => {
  const { panel, calls } = await openHeatmap(page, [
    capture({ prompt: 'A', fingerprintValue: '', }),
    capture({ prompt: 'B' }),
  ]);

  await expect(panel.getByRole('alert')).toContainText(/weight-derived model fingerprint/i);
  await expect(panel.getByText('AWAITING CAPTURE')).toBeVisible();
  expect(calls()).toBe(2);
});

test('Research Workflow refuses incompatible model fingerprints without rendering a heatmap', async ({ page }) => {
  const { panel, calls } = await openHeatmap(page, [
    capture({ prompt: 'A' }),
    capture({ prompt: 'B', fingerprintValue: 'sha256:' + 'b'.repeat(64) }),
  ]);

  await expect(panel.getByRole('alert')).toContainText(/incompatible activations/i);
  await expect(panel.getByText('AWAITING CAPTURE')).toBeVisible();
  expect(calls()).toBe(2);
});

test('Research Workflow refuses non-live or unattested tensor responses', async ({ page }) => {
  const { panel, calls } = await openHeatmap(page, [
    capture({ prompt: 'A', attested: false }),
    capture({ prompt: 'B' }),
  ]);

  await expect(panel.getByRole('alert')).toContainText(/weight-derived model fingerprint/i);
  await expect(panel.getByText('AWAITING CAPTURE')).toBeVisible();
  expect(calls()).toBe(2);
});

test('Research Workflow rejects incompatible residual dimensions', async ({ page }) => {
  const { panel, calls } = await openHeatmap(page, [
    capture({ prompt: 'A', matrix: [[1, 2], [3, 4]] }),
    capture({ prompt: 'B', matrix: [[1, 2, 3], [4, 5, 6]] }),
  ]);

  await expect(panel.getByRole('alert')).toContainText(/dimensions are incompatible/i);
  await expect(panel.getByText('AWAITING CAPTURE')).toBeVisible();
  expect(calls()).toBe(2);
});
