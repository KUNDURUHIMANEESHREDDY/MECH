# Baseline: Phase 0 pre-redesign test state (vitest + playwright)

- Branch: `redesign/visual-research-canvas`
- HEAD (`git rev-parse HEAD`): `c5d80c39ccbfbfff2fcc4623104f80554177ec4d`
- Date: 2026-08-07 15:14:47 +05:30
- Plan: `.omo/plans/redesign-ui-shell.md`

Gate measurement only — no fixes applied, no product-code changes. Only this
file was created.

---

## Suite 1: `npm run test:js` (Vitest)

- Command (workdir `frontend/`): `npm run test:js`
- Exit code: **0**
- Result: **all green** — 23 test files passed (23), 104 tests passed (104)
- Notes: only `act(...)` warnings on stderr for a few Visualization panel
  tests (SAEFeaturePanel, LogitLensViewer, ModelComparisonPanel); no failures.

Last ~20 output lines:

```
 ✓ tests/vitest/App.test.jsx (7 tests) 671ms
 ✓ tests/vitest/Sprint5Deliverable.test.jsx (2 tests) 6ms
 ✓ tests/vitest/Sprint3Platform.test.jsx (2 tests) 4ms
 ✓ tests/vitest/Sprint6Deliverable.test.jsx (1 test) 7ms
 ✓ tests/vitest/Sprint4Deliverable.test.jsx (3 tests) 6ms
 ✓ tests/vitest/DebuggerWorkflow.test.jsx (1 test) 94ms
 ✓ tests/vitest/DiscoveryMemoryModal.test.jsx (4 tests) 155ms
 ✓ tests/vitest/Settings.test.jsx (3 tests) 242ms
 ✓ tests/vitest/KnowledgeGraphView.test.jsx (3 tests) 145ms

 Test Files  23 passed (23)
      Tests  104 passed (104)
   Start at  15:13:02
   Duration  10.86s (transform 7.08s, setup 14.46s, collect 25.17s, tests 8.20s, environment 72.87s, prepare 8.47s)

EXIT_CODE=0
```

---

## Suite 2: `npx playwright test` (E2E)

- Command (workdir `frontend/`): `npx playwright test`
- Playwright config `frontend/playwright.config.js` DOES define a `webServer`
  key (`command: 'npm run dev:renderer'`, `url: http://localhost:5173`,
  `reuseExistingServer: !CI`), so the dev server was started/managed by
  Playwright itself (visible as `[WebServer]` output lines). No manual vite
  background process was started, so there is no PID to record as a cleanup
  receipt — Playwright owns the server lifecycle and shuts it down on exit.
- Exit code: **1**
- Result: **4 passed, 1 failed** (5 tests, 1 worker, ~1.1m)

### `shell-smoke.spec.js` per-test results

| # | Test | Result |
|---|------|--------|
| 1 | shell renders sidebar nav items (asserts first 10 of the 29 `nav-*` data-testids visible) | passed (3.1s) |
| 2 | each nav page renders and updates breadcrumb + hash (all 29 NAV_PAGES: header crumb label + `#key` hash) | passed (9.1s) |
| 3 | hash-only pages render when opened by URL (gpt2explorer / transformerExplorer) | passed (12.4s) |
| 4 | activity bar collapses and expands via its toggle button (`div.app.activity-collapsed` toggling via `header button[title*="activity bar"]`) | passed (2.0s) |
| 5 | topbar shows python status testid (`getByTestId('python-status')`) | **failed** (31.4s) |

### Failure evidence (exact, unpasteurized from run)

Test 5 `topbar shows python status testid` failed with a 30s test timeout:

```
Test timeout of 30000ms exceeded.

Error: expect(locator).toBeVisible() failed

Locator:  getByTestId('python-status')
Expected: visible
Received: undefined

Call log:
  - Expect "toBeVisible" with timeout 5000ms
  - waiting for getByTestId('python-status')
  - Protocol error (Runtime.callFunctionOn): Internal server error, session closed.

  104 | test('topbar shows python status testid', async ({ page }) => {
  105 |   await page.goto('/');
> 106 |   await expect(page.getByTestId('python-status')).toBeVisible();
      |                                                   ^
  107 | });
    at C:\Users\himaneeshreddyk\Downloads\MECH\frontend\tests\playwright\shell-smoke.spec.js:106:51
```

Final tally from runner:

```
  1 failed
    tests\playwright\shell-smoke.spec.js:104:1 › topbar shows python status testid
  4 passed (1.1m)
EXIT_CODE=1
```

### Interpretation (no fix attempted)

The pre-redesign shell does NOT surface a `python-status` data-testid in the
topbar — the locator resolved to `undefined` and the test timed out. This is
the one known pre-redesign red and is exactly the kind of gap Phase 0 is meant
to record. All other shell smoke checks (29 `nav-*` testids, header crumb
labels, hash routing, activity-bar toggle) pass on the current UI.

---

## Summary

| Suite | Command | Exit | Result |
|-------|---------|------|--------|
| Vitest | `npm run test:js` | 0 | 23 files / 104 tests, all green |
| Playwright E2E | `npx playwright test` | 1 | 4 passed, 1 failed (`python-status` testid timeout) |
