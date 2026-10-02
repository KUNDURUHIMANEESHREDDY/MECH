/**
 * Capture screenshots of the running MECH UI.
 *
 * Assumes:
 *   backend  http://127.0.0.1:8000  (python backend/main.py, PYTHONPATH=repo root)
 *   frontend http://localhost:5173   (npx vite)
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

const PAGES = [
  ["explorer", "01-model-explorer", 6000],
  ["transformer", "02-transformer-visualizer", 4500],
  ["network", "03-network", 4500],
  ["steering", "04-steering-lab", 4500],
  ["benchmark", "05-benchmark-dashboard", 4500],
  ["society", "06-research-society", 5500],
  ["workspace", "07-campaign-workspace", 4500],
  ["models", "08-models", 3000],
  ["settings", "09-settings", 2500],
  ["plugins", "10-plugin-sdk", 2500],
];

(async () => {
  fs.mkdirSync(OUT, { recursive: true });

  const consoleErrors = [];
  const pageErrors = [];

  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1,
  });
  const page = await ctx.newPage();

  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => pageErrors.push(String(e)));

  const captured = [];
  for (const [route, name, wait] of PAGES) {
    const url = `${BASE}/#${route}`;
    process.stdout.write(`-> ${route} ... `);
    try {
      await page.goto(url, { waitUntil: "networkidle", timeout: 45000 });
    } catch {
      await page.goto(url, { waitUntil: "domcontentloaded", timeout: 45000 });
    }
    await page.waitForTimeout(wait);
    const file = path.join(OUT, `${name}.png`);
    await page.screenshot({ path: file });
    captured.push(file);
    console.log(`saved ${name}.png`);
  }

  await browser.close();

  console.log(`\n${captured.length} screenshots -> ${path.relative(ROOT, OUT)}`);

  const uniq = (a) => [...new Set(a)];
  const ce = uniq(consoleErrors);
  const pe = uniq(pageErrors);
  console.log(`console errors: ${ce.length}   page errors: ${pe.length}`);
  ce.slice(0, 15).forEach((e) => console.log("  [console]", e.slice(0, 220)));
  pe.slice(0, 15).forEach((e) => console.log("  [page]", e.slice(0, 220)));

  fs.writeFileSync(
    path.join(OUT, "_console_errors.json"),
    JSON.stringify({ consoleErrors: ce, pageErrors: pe }, null, 2),
  );
})();
