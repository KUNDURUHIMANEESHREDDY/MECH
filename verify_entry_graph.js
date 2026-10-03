// Walk the Electron main-process require graph from the packaged entry point.
//
// Deleting unreferenced files needs evidence, not a grep. This resolves
// require()/import specifiers transitively from package.json's "main" and
// reports which files under frontend/electron and frontend/scripts are
// unreachable.
//
//   node verify_entry_graph.js
// Exit 0 always; prints the reachability report.
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "frontend");
const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, "package.json"), "utf8"));
const entry = path.join(ROOT, pkg.main);

const SCAN_DIRS = ["electron", "scripts"];
const EXTS = [".js", ".cjs", ".mjs", ".ts", ".jsx", ".tsx", ".vue", ".json"];

function candidates(fromFile, spec) {
  if (!spec.startsWith(".")) return [];
  const base = path.resolve(path.dirname(fromFile), spec);
  const out = [];
  if (fs.existsSync(base) && fs.statSync(base).isFile()) out.push(base);
  for (const ext of EXTS) {
    if (fs.existsSync(base + ext)) out.push(base + ext);
  }
  for (const ext of EXTS) {
    const idx = path.join(base, "index" + ext);
    if (fs.existsSync(idx)) out.push(idx);
  }
  return out;
}

function specifiers(file) {
  const src = fs.readFileSync(file, "utf8");
  const specs = [];
  const re = /(?:require\(\s*['"]([^'"]+)['"]\s*\)|from\s+['"]([^'"]+)['"]|import\s+['"]([^'"]+)['"])/g;
  let m;
  while ((m = re.exec(src)) !== null) {
    specs.push(m[1] || m[2] || m[3]);
  }
  return specs;
}

const seen = new Set();
const queue = [entry];

while (queue.length) {
  const file = queue.shift();
  if (seen.has(file)) continue;
  seen.add(file);
  let specs = [];
  try {
    specs = specifiers(file);
  } catch {
    continue;
  }
  for (const spec of specs) {
    for (const target of candidates(file, spec)) {
      if (!seen.has(target)) queue.push(target);
    }
  }
}

// Anything on disk in the scanned dirs that the graph never reached.
const onDisk = [];
function walk(dir) {
  for (const entryName of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entryName.name);
    if (entryName.isDirectory()) {
      if (["node_modules", "dist", "release"].includes(entryName.name)) continue;
      walk(full);
    } else if (/\.(js|cjs|mjs|ts|jsx|tsx|py|vue)$/.test(entryName.name)) {
      onDisk.push(full);
    }
  }
}
for (const d of SCAN_DIRS) {
  const dir = path.join(ROOT, d);
  if (fs.existsSync(dir)) walk(dir);
}

const reachable = onDisk.filter((f) => seen.has(f)).map((f) => path.relative(ROOT, f));
const unreachable = onDisk.filter((f) => !seen.has(f)).map((f) => path.relative(ROOT, f));

console.log("=".repeat(66));
console.log(`entry: ${path.relative(ROOT, entry)}`);
console.log("=".repeat(66));
// Files that are loaded without a require(): either named in an npm script, or
// referenced as a runtime path (e.g. window webPreferences.preload). Treating
// these as dead would invite deleting live code -- preload.js is exactly that.
const pkgScripts = Object.values(pkg.scripts || {}).join("\n");
function loadedOutOfBand(file) {
  const rel = path.relative(ROOT, file).split(path.sep).join("/");
  const base = path.basename(rel);
  if (pkgScripts.includes(base) || pkgScripts.includes(rel)) return "npm script";
  const all = [];
  for (const d of SCAN_DIRS) {
    const dir = path.join(ROOT, d);
    if (!fs.existsSync(dir)) continue;
    const stack = [dir];
    while (stack.length) {
      const cur = stack.pop();
      for (const e of fs.readdirSync(cur, { withFileTypes: true })) {
        const full = path.join(cur, e.name);
        if (e.isDirectory()) stack.push(full);
        else if (/\.(js|ts|json)$/.test(e.name)) {
          try { all.push(fs.readFileSync(full, "utf8")); } catch {}
        }
      }
    }
  }
  if (all.some((src) => src.includes(base))) return "referenced by path/string";
  return null;
}

console.log(`\nREACHABLE via require() (${reachable.length}):`);
for (const f of reachable.sort()) console.log(`  ${f}`);

const outOfBand = [];
const orphans = [];
for (const f of unreachable) {
  const why = loadedOutOfBand(f);
  if (why) outOfBand.push([path.relative(ROOT, f), why]);
  else orphans.push(path.relative(ROOT, f));
}

console.log(`\nLOADED OUT OF BAND, not via require() (${outOfBand.length}):`);
console.log("  (still live -- do not delete)");
for (const [f, why] of outOfBand.sort()) console.log(`  ${f}   [${why}]`);

console.log(`\nORPHANED (${orphans.length}):`);
console.log("  (on disk, not loaded from the entry point, and not named anywhere)");
for (const f of orphans.sort()) console.log(`  ${f}`);

const tsconfigInclude = JSON.parse(
  fs.readFileSync(path.join(ROOT, "tsconfig.json"), "utf8"),
).include;
console.log(`\nnote: tsconfig include = ${JSON.stringify(tsconfigInclude)}`);
console.log(
  "      so TypeScript outside src/ is never compiled, and cannot run. " +
  "A .ts file listed as orphaned stays inert unless a build step is added.",
);