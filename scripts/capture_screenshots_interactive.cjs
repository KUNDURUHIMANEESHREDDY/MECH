/**
 * Drive the running MECH UI through real interactions and capture screenshots.
 *
 * Assumes:
 *   backend  http://127.0.0.1:8000  (python backend/main.py, PYTHONPATH=repo root)
 *   frontend http://localhost:5173   (npx vite)
 *
 * Unlike capture_screenshots.cjs this loads the model, runs a prompt and runs a
 * steering pass so the captured views contain measured results, not empty states.
 *
 * Writes PNGs into docs/images/ui/.
 */
const fs = require("fs");
const path = require("path");

const { chromium } = require(
  path.join(__dirname, "..", "frontend", "node_modules", "playwright"),
);

const BASE = "http://localhost:5173";
const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "docs", "images", "ui");

const consoleErrors = [];
const pageErrors = [];

/** Click the first element matching a visible text selector. */
async function clickText(page, selector, text, timeout = 8000) {
  const loc = page.locator(selector, { hasText: text }).first();
  try {
    await loc.waitFor({ state: "visible", timeout });
    await loc.click();
    return true;
  } catch {
    return false;
  }
}

async function shot(page, name) {
  const file = path.join(OUT, `${name}.png`);
  await page.screenshot({ path: file });
  console.log(`   saved ${name}.png`);
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });

  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
  const page = await ctx.newPage();
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => pageErrors.push(String(e)));

  // ---------------------------------------------------------------- explorer
  console.log("-> Model Explorer: load model + run prompt");
  await page.goto(`${BASE}/#explorer`, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(4000);

  const loaded = await clickText(page, "button", "Load model", 12000);
  console.log(`   load model clicked: ${loaded}`);
  await page.waitForTimeout(9000);

  const ran = await clickText(page, "button", "Run prompt", 12000);
  console.log(`   run prompt clicked: ${ran}`);
  await page.waitForTimeout(14000);

  await page.mouse.wheel(0, 700);
  await page.waitForTimeout(2500);
  await shot(page, "01-model-explorer");

  await page.mouse.wheel(0, 900);
  await page.waitForTimeout(2500);
  await shot(page, "01b-model-explorer-inspectors");

  // -------------------------------------------------------------- transformer
  console.log("-> Transformer Visualizer: run prompt");
  await page.goto(`${BASE}/#transformer`, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(4000);
  await clickText(page, "button", "Run prompt", 12000);
  await page.waitForTimeout(14000);
  await page.mouse.wheel(0, 800);
  await page.waitForTimeout(2500);
  await shot(page, "02-transformer-visualizer");

  // ------------------------------------------------------------------ network
  console.log("-> Network view");
  await page.goto(`${BASE}/#network`, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(11000);
  await shot(page, "03-network");

  // ------------------------------------------------------------------ steering
  console.log("-> Steering Lab: run a steer");
  await page.goto(`${BASE}/#steering`, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(4000);

  // Point the contrast pair at the France/Paris prompt so a flip is reachable.
  // The three prompt fields are the first three text inputs in the form.
  const textInputs = page.locator('input[type="text"]');
  const n = await textInputs.count();
  console.log(`   text inputs found: ${n}`);
  if (n >= 3) {
    await textInputs.nth(0).fill("The capital of France is");
    await textInputs.nth(1).fill("The capital of France is");
    await textInputs.nth(2).fill("The capital of Japan is");
  }

  // Strength 40 at layer 10 is where this contrast pair actually flips.
  const alpha = page.locator('input[type="number"]').first();
  try {
    await alpha.fill("40");
  } catch {}
  const layerSelect = page.locator("select").first();
  try {
    await layerSelect.selectOption("10");
  } catch {
    try {
      await layerSelect.selectOption({ label: "L10" });
    } catch {}
  }
  await page.waitForTimeout(1200);

  const steered = await clickText(page, 'button[type="submit"]', "Steer", 12000);
  console.log(`   steer clicked: ${steered}`);
  await page.waitForTimeout(20000);
  await shot(page, "04-steering-lab");

  // ----------------------------------------------------------------- society
  console.log("-> Research Society: run a workflow");
  await page.goto(`${BASE}/#society`, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(4000);
  const goal = page.locator("input[type='text'], textarea").first();
  try {
    await goal.fill("Reproduce IOI on gpt2-small and find causally important heads");
  } catch {}
  const started = await clickText(page, "button", "Run Society", 12000);
  console.log(`   run society clicked: ${started}`);
  await page.waitForTimeout(30000);
  await shot(page, "06-research-society");

  await page.waitForTimeout(25000);
  await shot(page, "06b-research-society-progress");

  // --------------------------------------------------------------- benchmark
  console.log("-> Benchmark dashboard");
  await page.goto(`${BASE}/#benchmark`, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(5000);
  await clickText(page, "button", "Run", 10000);
  await page.waitForTimeout(15000);
  await shot(page, "05-benchmark-dashboard");

  await browser.close();

  const uniq = (a) => [...new Set(a)];
  const ce = uniq(consoleErrors);
  const pe = uniq(pageErrors);
  console.log(`\nconsole errors: ${ce.length}   page errors: ${pe.length}`);
  ce.slice(0, 12).forEach((e) => console.log("  [console]", e.slice(0, 200)));
  pe.slice(0, 12).forEach((e) => console.log("  [page]", e.slice(0, 200)));

  fs.writeFileSync(
    path.join(OUT, "_console_errors.json"),
    JSON.stringify({ consoleErrors: ce, pageErrors: pe }, null, 2),
  );
})();
