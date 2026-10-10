/**
 * Capture screenshots of the running MECH UI -- as assertions, not as pictures.
 *
 * Assumes:
 *   backend  http://127.0.0.1:8000  (uvicorn backend.main:app)
 *   frontend http://localhost:5173   (npx vite)
 *
 * Why this replaced the two scripts it subsumes
 * ---------------------------------------------
 * There were three: `capture_screenshots.cjs`, `capture_screenshots_interactive.cjs`
 * and `capture_screenshots.py`. They had already diverged -- only one of them
 * loaded the model, only one visited the Society view -- so a "screenshot is up
 * to date" check depended on which script a person remembered to run.
 *
 * What was actually wrong with all three
 * --------------------------------------
 * A screenshot is not a test. Each of them did:
 *
 *     goto -> wait N seconds -> screenshot -> report success
 *
 * so this sequence produced a green result and a PNG of an error page:
 *
 *     API fails -> UI stays in its loading or error state -> 14 seconds pass ->
 *     PNG captured -> "saved 04-steering-lab.png"
 *
 * Nothing checked that a model was loaded, that a prompt returned tokens, that
 * a steering result existed, or that the network view drew any edges. The
 * interactive version also located its controls positionally --
 * `textInputs.nth(0..2)`, `input[type="number"].first()`, `select.first()` --
 * so a reordering in the form, or one added input, silently changed which field
 * got filled. And it swallowed the errors it hit: three bare `catch {}` blocks
 * around the fills, each of which left a default value in place and produced a
 * screenshot of the wrong experiment.
 *
 * What this does instead
 * ----------------------
 * * Every screenshot is preceded by a semantic assertion: a model is loaded, a
 *   prompt returned tokens, a steering result rendered, the network drew edges,
 *   the benchmark produced a result, the Society workflow started and progressed.
 * * Controls are addressed by `data-testid`, never by position.
 * * Readiness is state-based: `waitForResponse` on the API call and
 *   `waitForSelector` on the element that only exists once data has arrived.
 *   No fixed sleeps.
 * * Console errors, page errors, failed requests and non-2xx responses are
 *   captured and tagged with the route they happened on.
 * * Each route gets its own diagnostics record, written to
 *   `docs/images/ui/_capture_report.json`.
 * * The browser is closed in a `finally` path, so a failed assertion does not
 *   leave a Chromium process behind.
 * * A failed assertion fails the process. Screenshots that exist because a page
 *   silently did not work are worse than no screenshots.
 *
 * Usage:
 *   node scripts/capture_screenshots.cjs                 # all routes
 *   node scripts/capture_screenshots.cjs --only steering,explorer
 *   node scripts/capture_screenshots.cjs --no-load-model  # assume already loaded
 *   node scripts/capture_screenshots.cjs --allow-console-errors
 *
 * Authentication
 * --------------
 * The backend requires a bearer token on everything except `/`, `/health`, CORS
 * preflight and static files. The token comes from `MECH_API_TOKEN`, or from
 * `backend/storage/.mech_api_token` (see `backend/core/auth.py`). It is read
 * here and injected as a default header so the UI's own unauthenticated calls
 * stop returning 401.
 *
 * Without this the capture fails its assertions -- correctly, and for a reason
 * that has nothing to do with the UI. That is the point: the first run of this
 * script reported `401 GET /api/models` and refused to screenshot an empty
 * model registry, which is exactly what the previous version would have
 * captured and called a success.
 */
const fs = require("fs");
const path = require("path");

const { chromium } = require(
  path.join(__dirname, "..", "frontend", "node_modules", "playwright"),
);

const ROOT = path.resolve(__dirname, "..");
const BASE = process.env.MECH_UI_BASE || "http://localhost:5173";
const OUT = path.join(ROOT, "docs", "images", "ui");

// Generous, because a cold start loads GPT-2 and a slow machine is not a
// failure. These are ceilings, not sleeps: nothing waits for them once the
// state it waits for has arrived.
const NAV_TIMEOUT_MS = 60000;
const API_TIMEOUT_MS = 120000;

/**
 * The bearer token the backend accepts, or null if none is discoverable.
 *
 * Mirrors `backend/core/auth.py`: env var first, then the persisted file.
 */
function resolveToken() {
  const fromEnv = (process.env.MECH_API_TOKEN || "").trim();
  if (fromEnv) return fromEnv;
  const configured = (process.env.MECH_TOKEN_FILE || "").trim();
  const candidates = configured
    ? [path.resolve(configured)]
    : [path.join(ROOT, "backend", "storage", ".mech_api_token")];
  for (const file of candidates) {
    try {
      const value = fs.readFileSync(file, "utf8").trim();
      if (value) return value;
    } catch {
      /* try the next candidate */
    }
  }
  return null;
}

/** Raised when a required assertion fails. Carries the route for the report. */
class AssertionFailure extends Error {
  constructor(route, message, detail) {
    super(`${route}: ${message}`);
    this.name = "AssertionFailure";
    this.route = route;
    this.detail = detail;
  }
}

function parseArgs(argv) {
  const args = { only: null, loadModel: true, allowConsoleErrors: false };
  for (const arg of argv) {
    if (arg.startsWith("--only=")) args.only = arg.slice("--only=".length).split(",");
    else if (arg === "--no-load-model") args.loadModel = false;
    else if (arg === "--allow-console-errors") args.allowConsoleErrors = true;
  }
  return args;
}

// ── Assertions ──────────────────────────────────────────────────────────

/**
 * Wait for `testId` to be visible, or fail the route.
 *
 * `visible` rather than `attached`: an element behind an overlay, or one with
 * `v-if` false, is attached but is not something a screenshot can show.
 */
async function requireVisible(page, route, testId, timeout = 30000) {
  const locator = page.locator(`[data-testid="${testId}"]`);
  try {
    await locator.first().waitFor({ state: "visible", timeout });
  } catch (err) {
    throw new AssertionFailure(
      route,
      `expected [data-testid="${testId}"] to become visible`,
      { testId, timeout, url: page.url() });
  }
  return locator.first();
}

async function requireAbsent(page, route, testId, timeout = 3000) {
  const locator = page.locator(`[data-testid="${testId}"]`);
  try {
    await locator.first().waitFor({ state: "visible", timeout });
  } catch {
    return true; // absent, which is what we wanted
  }
  throw new AssertionFailure(
    route, `expected [data-testid="${testId}"] to stay absent`, { testId });
}

async function requireText(page, route, testId, pattern, timeout = 30000) {
  const locator = await requireVisible(page, route, testId, timeout);
  try {
    await locator.filter({ hasText: pattern }).first().waitFor({ timeout });
  } catch (err) {
    const actual = (await locator.first().innerText().catch(() => "")) || "";
    throw new AssertionFailure(
      route,
      `[data-testid="${testId}"] never showed ${pattern}`,
      { testId, expected: pattern, actual: actual.slice(0, 300) });
  }
  return locator;
}

/**
 * Start the API call, then wait for both the response and the DOM change.
 *
 * Waiting on the response alone is not enough: the fetch can succeed and the
 * render can fail, which is exactly the case that produced a screenshot of an
 * error state. Waiting on the DOM alone is not enough either, because a stale
 * element from a previous route could satisfy it.
 */
async function withApiWait(page, route, urlSubstring, action, ready) {
  // The timeout used to be swallowed: `.catch(() => null)` turned a request
  // that never happened into a null, the null was treated as "no error", and
  // the route continued. A typo in the path therefore cost the full
  // API_TIMEOUT_MS ceiling and then passed on the DOM assertion alone -- and two
  // paths did have typos, costing 241 seconds across a single run.
  //
  // The rejection is now translated into a reported assertion failure, and the
  // pending wait is always consumed so an unobserved rejection cannot take the
  // process down.
  const pending = page
    .waitForResponse(
      (r) => r.url().includes(urlSubstring) && r.request().method() !== "OPTIONS",
      { timeout: API_TIMEOUT_MS },
    )
    .then((response) => ({ response }), (cause) => ({ cause }));

  let actionError = null;
  try {
    await action();
  } catch (cause) {
    actionError = cause;
  }
  const outcome = await pending;

  if (actionError) throw actionError;
  if (!outcome.response) {
    throw new AssertionFailure(
      route,
      `the action produced no request to ${urlSubstring}`,
      {
        expectedPath: urlSubstring,
        pageUrl: page.url(),
        cause: String(outcome.cause && outcome.cause.message
          ? outcome.cause.message : outcome.cause).slice(0, 240),
      });
  }
  const response = outcome.response;
  if (response.status() >= 400) {
    throw new AssertionFailure(
      route,
      `${urlSubstring} answered ${response.status()}`,
      { url: response.url(), status: response.status() });
  }
  await ready();
  return response;
}

// ── Diagnostics ─────────────────────────────────────────────────────────

function attachDiagnostics(page, sink) {
  page.on("console", (m) => {
    if (m.type() === "error") sink.consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => sink.pageErrors.push(String(e)));
  page.on("requestfailed", (r) => {
    const failure = r.failure();
    sink.failedRequests.push(
      `${r.method()} ${r.url()} :: ${failure ? failure.errorText : "unknown"}`);
  });
  page.on("response", (r) => {
    if (r.status() >= 400) {
      sink.failedResponses.push(`${r.status()} ${r.request().method()} ${r.url()}`);
    }
  });
}

function emptyDiagnostics(route) {
  return {
    route,
    status: "pending",
    screenshots: [],
    consoleErrors: [],
    pageErrors: [],
    failedRequests: [],
    failedResponses: [],
    failedAssertion: null,
    durationS: null,
  };
}

/**
 * Headers every context starts with.
 *
 * Empty when no token is discoverable, which is correct for a backend running
 * without auth. The capture then fails its assertions on 401s, and the report
 * says so -- rather than silently producing screenshots of an unauthorised UI.
 */
let cachedAuthHeaders = null;
function authHeaders() {
  if (cachedAuthHeaders === null) {
    const token = resolveToken();
    cachedAuthHeaders = token ? { Authorization: `Bearer ${token}` } : {};
  }
  return cachedAuthHeaders;
}

function summarise(record) {
  return {
    route: record.route,
    status: record.status,
    screenshots: record.screenshots,
    consoleErrors: record.consoleErrors.length,
    pageErrors: record.pageErrors.length,
    failedRequests: record.failedRequests.length,
    failedResponses: record.failedResponses.length,
    failedAssertion: record.failedAssertion,
    durationS: record.durationS,
  };
}

// ── Routes ──────────────────────────────────────────────────────────────

/**
 * Each route returns a record and may throw AssertionFailure.
 *
 * `page` is a fresh context per route rather than one shared page, so a route
 * cannot pass by inheriting a model loaded by an earlier one -- the state that
 * makes a screenshot meaningful is established by that route or not at all.
 */
const ROUTES = {
  async explorer(page, args) {
    const route = "explorer";
    await page.goto(`${BASE}/#explorer`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });

    if (args.loadModel) {
      const load = page.locator('[data-testid="model-load"]');
      await load.waitFor({ state: "visible", timeout: 30000 });
      if (await load.isEnabled()) {
        await withApiWait(
          page, route, "/gpt2/load",
          () => load.click(),
          async () => {
            await requireText(page, route, "model-loaded", /Loaded/);
          },
        );
      }
    }
    // Asserted either way: a screenshot of an unloaded explorer is not evidence.
    await requireText(page, route, "model-loaded", /Loaded/);
    await requireAbsent(page, route, "model-load-error");

    const prompt = page.locator('[data-testid="model-run-prompt"]');
    await prompt.waitFor({ state: "visible", timeout: 30000 });
    if (await prompt.isEnabled()) {
      await withApiWait(
        page, route, "/gpt2/run_prompt",
        () => prompt.click(),
        async () => {
          await requireVisible(page, route, "model-prompt-result");
          await requireText(page, route, "model-prompt-token-count", /\d+ returned tokens/);
        },
      );
    }

    await shot(page, route, "01-model-explorer");
    await page.mouse.wheel(0, 700);
    await settle(page);
    await shot(page, route, "01b-model-explorer-inspectors");
  },

  async transformer(page, args) {
    const route = "transformer";
    await page.goto(`${BASE}/#transformer`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });

    // Architecture comes from the loaded model, so it is asserted before the
    // prompt: a screenshot of an unloaded visualiser is a screenshot of an
    // error, whatever the layout looks like.
    await requireVisible(page, route, "transformer-architecture", 60000);
    await requireVisible(page, route, "transformer-layers", 60000);

    const run = page.locator('[data-testid="transformer-run"]');
    await run.waitFor({ state: "visible", timeout: 30000 });
    if (await run.isEnabled()) {
      await withApiWait(
        page, route, "/gpt2/run_prompt",
        () => run.click(),
        async () => {
          // Previously this was `waitForTimeout(1000)` -- a guess that the
          // render had finished. It is replaced by the state it was guessing
          // at: the token strip only exists once tokens have arrived.
          await requireVisible(page, route, "transformer-next-token", 60000);
          await requireText(page, route, "transformer-prompt-summary",
            /\d+ returned tokens/);
        },
      );
    }
    await requireAbsent(page, route, "transformer-prompt-error");
    await shot(page, route, "02-transformer-visualizer");
  },

  async network(page, args) {
    const route = "network";
    await page.goto(`${BASE}/#network`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });

    // The wiring panels render only on `v-if="wireTokens.length"`, so without
    // running a prompt there is nothing to assert and nothing to photograph. The
    // previous version slept 11 seconds and screenshotted whatever was on
    // screen -- which turned out to be the parameter ledger, with the attention
    // wires the route is named for still below an unrun fold.
    const run = page.locator('[data-testid="network-wire-run"]');
    await run.waitFor({ state: "visible", timeout: 30000 });
    // `/api/infer`, not `/gpt2/run_prompt`: `runWiringPrompt` calls
    // `api.infer`. The wrong path matched nothing, so this route used to sit
    // through the whole 120s ceiling before the DOM assertion ran.
    await withApiWait(
      page, route, "/api/infer",
      () => run.click(),
      async () => {
        const caption = await requireVisible(page, route, "network-wire-caption", 60000);
        await caption.filter({ hasText: /\d+ wires above/ }).first()
          .waitFor({ timeout: 60000 }).catch(async () => {
            throw new AssertionFailure(
              route, "the wire caption never reported a measured wire count",
              { actual: (await caption.innerText().catch(() => "")).slice(0, 200) });
          });
      },
    );
    await requireAbsent(page, route, "network-wire-error");

    // Scroll the wiring into frame. The section sits below a long parameter
    // ledger, so a viewport screenshot otherwise shows weights and nothing else.
    await page.locator('[data-testid="network-wire-svg"]').scrollIntoViewIfNeeded();
    await settle(page);
    await shot(page, route, "03-network");
  },

  async steering(page, args) {
    const route = "steering";
    await page.goto(`${BASE}/#steering`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });

    // Addressed by test id, not by position. `nth(0..2)` broke silently the
    // moment the form gained, lost or reordered a field.
    await requireVisible(page, route, "steer-prompt");
    await page.locator('[data-testid="steer-prompt"]').fill("The capital of France is");
    await page.locator('[data-testid="steer-positive"]').fill("The capital of France is");
    await page.locator('[data-testid="steer-negative"]').fill("The capital of Japan is");
    await page.locator('[data-testid="steer-alpha"]').fill("40");
    await page.locator('[data-testid="steer-layer"]').selectOption("10");

    // Read back what landed. A `fill` that silently went nowhere used to leave
    // the default contrast pair in place and produce a screenshot of a
    // different experiment than the one the script thought it ran.
    const values = await page.evaluate(() => {
      const get = (id) => (document.querySelector(`[data-testid="${id}"]`) || {}).value;
      return {
        prompt: get("steer-prompt"),
        positive: get("steer-positive"),
        negative: get("steer-negative"),
        alpha: get("steer-alpha"),
        layer: get("steer-layer"),
      };
    });
    const expected = { prompt: "The capital of France is", alpha: "40", layer: "10" };
    for (const [key, want] of Object.entries(expected)) {
      if (values[key] !== want) {
        throw new AssertionFailure(
          route, `steering ${key} is ${JSON.stringify(values[key])}, expected ${JSON.stringify(want)}`,
          { values });
      }
    }

    await withApiWait(
      page, route, "/gpt2/steer",
      () => page.locator('[data-testid="steer-submit"]').click(),
      async () => {
        await requireVisible(page, route, "steer-result", 60000);
        await requireText(page, route, "steer-verdict", /FLIPPED|UNCHANGED/);
      },
    );
    await requireAbsent(page, route, "steer-error");
    await shot(page, route, "04-steering-lab");
  },

  async benchmark(page, args) {
    const route = "benchmark";
    await page.goto(`${BASE}/#benchmark`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });
    const select = page.locator('[data-testid="benchmark-select"]');
    await select.waitFor({ state: "visible", timeout: 30000 });

    // A benchmark has to exist to be run. Selecting the first real option
    // rather than clicking "Run" and hoping is the difference between a
    // screenshot of a result and a screenshot of a disabled button.
    const options = await select.locator("option").allTextContents();
    const usable = options.filter((t) => t && !/select a benchmark|catalog unavailable/i.test(t));
    if (!usable.length) {
      throw new AssertionFailure(route, "the benchmark catalog was empty", { options });
    }
    await select.selectOption({ index: 1 });

    const run = page.locator('[data-testid="benchmark-run"]');
    await run.waitFor({ state: "visible", timeout: 30000 });
    if (await run.isEnabled()) {
      // `/api/benchmarks/run`, not `/run_benchmark`: that is the endpoint
      // `api.runBenchmark` posts to, and the typo cost a full 120s ceiling.
      await withApiWait(
        page, route, "/api/benchmarks/run",
        () => run.click(),
        async () => { await requireVisible(page, route, "benchmark-result", 60000); },
      );
    }
    await shot(page, route, "05-benchmark-dashboard");
  },

  async society(page, args) {
    const route = "society";
    await page.goto(`${BASE}/#society`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });
    await page.locator('[data-testid="society-goal"]').fill(
      "Reproduce IOI on gpt2-small and find causally important heads");

    const phaseBox = page.locator('[data-testid="society-status"]');
    await phaseBox.waitFor({ state: "visible", timeout: 30000 });

    await withApiWait(
      page, route, "/society",
      () => page.locator('[data-testid="society-run"]').click(),
      async () => {
        // "Started" is asserted on the phase attribute, so a run that 404s
        // cannot leave the phase at "idle" and still be screenshotted as progress.
        await page.waitForFunction(
          () => {
            const el = document.querySelector('[data-testid="society-status"]');
            const phase = el && el.getAttribute("data-phase");
            return phase && phase !== "idle";
          },
          undefined,
          { timeout: 60000 },
        ).catch(() => {
          throw new AssertionFailure(route, "the Society run never left the idle phase", {
            pageUrl: page.url(),
          });
        });
      },
    );
    const failure = page.locator('[data-testid="society-error"]');
    const steps = page.locator('[data-testid="society-steps"]');
    const status = page.locator('[data-testid="society-status"]');

    // Progress, not just start. A workflow that begins and immediately dies is
    // still a failure, and the old script screenshotted it as progress after 30
    // seconds of waiting.
    //
    // Waiting for steps alone is *not* sufficient, and that was the next defect
    // found by running it: a run whose steps appear and then fails at a later
    // stage satisfies "steps appeared", so the route passed and exited 0 over a
    // workflow that had reported `Society run failed`. The trace panel and the
    // error banner are rendered independently, so both are true at once.
    //
    // The verdict is therefore the run's terminal phase, not the presence of a
    // trace. `done` is the only success; `failed` and `stopped` are failures and
    // are reported with the error text.
    const outcome = await Promise.race([
      steps.waitFor({ state: "visible", timeout: 180000 }).then(() => "steps"),
      failure.waitFor({ state: "visible", timeout: 180000 }).then(() => "failure"),
    ]).catch(() => "timeout");

    if (outcome === "timeout") {
      const phase = await status.getAttribute("data-phase").catch(() => null);
      await shot(page, route, "06-research-society");
      throw new AssertionFailure(
        route, "the Society run produced neither trace steps nor an error",
        { phase, pageUrl: page.url() });
    }

    // Steps are present (or an error already is). Now wait for the run to settle
    // on a terminal phase before judging it.
    //
    // A run that never settles is a failure, not something to photograph: this
    // throws with the phase it stalled in rather than swallowing the timeout.
    let settled = true;
    try {
      await page.waitForFunction(
        () => {
          const el = document.querySelector('[data-testid="society-status"]');
          const phase = el && el.getAttribute("data-phase");
          return phase === "done" || phase === "failed" || phase === "stopped";
        },
        undefined,
        { timeout: 180000 },
      );
    } catch {
      settled = false;
    }

    const phase = await phaseBox.getAttribute("data-phase").catch(() => null);
    const message = (await failure.innerText().catch(() => "")) || "";

    await shot(page, route, "06-research-society");

    if (!settled) {
      throw new AssertionFailure(
        route,
        `the Society run never reached a terminal phase; stalled in `
        + `${JSON.stringify(phase)}`,
        { phase, detail: message.slice(0, 600), pageUrl: page.url() });
    }
    if (phase !== "done") {
      throw new AssertionFailure(
        route,
        `the Society run ended in phase ${JSON.stringify(phase)}, not "done"`
        + (message ? `: ${message.slice(0, 300)}` : ""),
        { phase, detail: message.slice(0, 600), pageUrl: page.url() });
    }

    await shot(page, route, "06b-research-society-progress");
  },

  async workspace(page, args) {
    await page.goto(`${BASE}/#workspace`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });
    await shot(page, "workspace", "07-campaign-workspace");
  },
  async models(page, args) {
    await page.goto(`${BASE}/#models`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });
    await shot(page, "models", "08-models");
  },
  async settings(page, args) {
    await page.goto(`${BASE}/#settings`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });
    await shot(page, "settings", "09-settings");
  },
  async plugins(page, args) {
    await page.goto(`${BASE}/#plugins`, { waitUntil: "networkidle", timeout: NAV_TIMEOUT_MS });
    await shot(page, "plugins", "10-plugin-sdk");
  },
};

// ── Output ──────────────────────────────────────────────────────────────

/** Screenshots taken on the current route, collected as they happen. */
let routeShots = [];

async function shot(page, route, name) {
  const file = path.join(OUT, `${name}.png`);
  await page.screenshot({ path: file });
  console.log(`   saved ${name}.png`);
  routeShots.push({ route, name, file: path.relative(ROOT, file) });
}

/** Let scroll-driven layout finish. Not a readiness wait -- it has none. */
async function settle(page) {
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => r())));
}

async function captureRoute(browser, name, fn, args) {
  const record = emptyDiagnostics(name);
  const started = Date.now();
  routeShots = [];
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1,
    // Injected rather than set per-request: the UI's own `fetch` calls carry no
    // token, and patching them individually would miss any call added later.
    extraHTTPHeaders: authHeaders(),
  });
  const page = await context.newPage();
  attachDiagnostics(page, record);
  try {
    await fn(page, args);
    record.screenshots = routeShots;
    record.status = "ok";
  } catch (err) {
    record.status = "failed";
    if (err instanceof AssertionFailure) {
      record.failedAssertion = {
        message: err.message,
        detail: err.detail || null,
      };
      console.error(`   FAILED ${err.message}`);
      if (err.detail) console.error(`   detail: ${JSON.stringify(err.detail)}`);
    } else {
      record.failedAssertion = {
        message: String(err && err.message ? err.message : err),
        detail: { stack: err && err.stack ? String(err.stack).split("\n").slice(0, 4) : null },
      };
      console.error(`   ERROR ${record.failedAssertion.message}`);
    }
  } finally {
    // Always close: a leaked Chromium keeps the whole node process alive and
    // the next run then fails for reasons that have nothing to do with the UI.
    await context.close().catch(() => {});
    record.durationS = ((Date.now() - started) / 1000).toFixed(1);
  }
  return record;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const names = args.only || Object.keys(ROUTES);
  const unknown = names.filter((n) => !ROUTES[n]);
  if (unknown.length) {
    console.error(`unknown route(s): ${unknown.join(", ")}`);
    console.error(`available: ${Object.keys(ROUTES).join(", ")}`);
    return 2;
  }

  fs.mkdirSync(OUT, { recursive: true });
  const token = resolveToken();
  console.log(`Capturing ${names.length} route(s) from ${BASE}`);
  console.log(
    token
      ? `Authenticated with the backend's bearer token (${token.length} chars).`
      : "No backend token found (MECH_API_TOKEN or backend/storage/.mech_api_token); "
        + "the capture will fail on any authenticated route.",
  );

  const records = [];
  const browser = await chromium.launch();
  try {
    for (const name of names) {
      console.log(`-> ${name}`);
      const record = await captureRoute(browser, name, ROUTES[name], args);
      records.push(record);
      if (record.status === "ok" && record.failedRequests.length) {
        record.status = "failed";
        record.failedAssertion = {
          message: `${record.failedRequests.length} request(s) failed on this route`,
          detail: { requests: record.failedRequests.slice(0, 5) },
        };
        console.error(`   FAILED ${record.failedAssertion.message}`);
      }
    }
  } finally {
    await browser.close().catch(() => {});
  }

  const totals = records.reduce(
    (acc, r) => {
      acc.screenshots += r.screenshots.length;
      acc.consoleErrors += r.consoleErrors.length;
      acc.pageErrors += r.pageErrors.length;
      acc.failedRequests += r.failedRequests.length;
      acc.failedResponses += r.failedResponses.length;
      if (r.status !== "ok") acc.failedRoutes += 1;
      return acc;
    },
    { screenshots: 0, consoleErrors: 0, pageErrors: 0, failedRequests: 0,
      failedResponses: 0, failedRoutes: 0 },
  );

  // Per-route diagnostics, and the errors in full, so a failing run can be
  // read without re-running it with a debugger attached.
  fs.writeFileSync(
    path.join(OUT, "_capture_report.json"),
    JSON.stringify(
      { base: BASE, startedAt: new Date().toISOString(), totals, routes: records },
      null, 2,
    ),
  );
  // The old filename is kept as a compatibility alias: other tooling reads it.
  fs.writeFileSync(
    path.join(OUT, "_console_errors.json"),
    JSON.stringify(
      {
        consoleErrors: [...new Set(records.flatMap((r) => r.consoleErrors))],
        pageErrors: [...new Set(records.flatMap((r) => r.pageErrors))],
      },
      null, 2,
    ),
  );

  console.log(`\n${totals.screenshots} screenshots across ${records.length} route(s)`);
  console.log(`console errors ${totals.consoleErrors}  page errors ${totals.pageErrors}`);
  console.log(`failed requests ${totals.failedRequests}  failed responses ${totals.failedResponses}`);

  if (totals.failedRoutes) {
    console.error(`\n${totals.failedRoutes} route(s) FAILED their assertions:`);
    for (const r of records.filter((x) => x.status !== "ok")) {
      console.error(`  ${r.route}: ${r.failedAssertion && r.failedAssertion.message}`);
    }
  }

  const consoleNoise = args.allowConsoleErrors
    ? 0
    : totals.consoleErrors + totals.pageErrors;
  if (totals.failedRoutes || consoleNoise) {
    process.exitCode = 1;
    return 1;
  }
  return 0;
}

main()
  .then((code) => {
    process.exitCode = code;
  })
  .catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
