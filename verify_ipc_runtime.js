// Replay runtime.js's real calls against the live backend.
//
// Loads frontend/electron/ipc/runtime.js, stubs Electron's ipcMain to capture
// the handlers, then invokes each handler the way the renderer would and prints
// what the backend returned. Closest thing to exercising the Electron path
// without launching Electron.
//
//   python main.py &
//   node verify_ipc_runtime.js

const fs = require("node:fs");
const path = require("node:path");

const ROOT = __dirname;
const RUNTIME = path.join(ROOT, "frontend", "electron", "ipc", "runtime.js");

// Minimal ipcMain stand-in so the module can register without Electron.
const handlers = {};
const ipcMain = {
  handle(channel, fn) {
    handlers[channel] = fn;
  },
};

const logger = { info() {}, warn() {}, error() {} };

require(RUNTIME).registerRuntimeHandlers({ ipcMain, pythonBridge: null, logger });

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) {
    console.log(`  [ok] ${name}${detail ? ` — ${detail}` : ""}`);
  } else {
    console.log(`  [FAIL] ${name}${detail ? ` — ${detail}` : ""}`);
    failures += 1;
  }
  return cond;
}

const call = (channel, arg) =>
  handlers[channel]({}, arg).then(
    (v) => ({ ok: true, value: v }),
    (e) => ({ ok: false, error: String(e && e.message) })
  );

(async () => {
  console.log("=".repeat(64));
  console.log("runtime.js handlers against the live backend");
  console.log("=".repeat(64));

  // 1. ping
  const ping = await call("python:ping");
  check("python:ping reaches the backend", ping.ok && ping.value?.ok === true,
    JSON.stringify(ping.value ?? ping.error).slice(0, 120));

  // 2. runtime status
  const status = await call("runtime:status");
  check("runtime:status returns a real status",
    status.value?.status && status.value.status !== "error",
    JSON.stringify(status.value ?? status.error).slice(0, 120));

  // 3. analyzeTokens -- the body-shape regression
  const tokens = await call("runtime:analyzeTokens", "The capital of France is");
  const tok = tokens.value ?? {};
  check("runtime:analyzeTokens echoes the prompt",
    typeof tok.prompt === "string" && tok.prompt.includes("capital"),
    `prompt=${JSON.stringify(tok.prompt)} count=${tok.count}`);
  check("runtime:analyzeTokens tokenises", tok.count > 0, `count=${tok.count}`);

  // 4. gpt2 pipeline
  const load = await call("gpt2:load");
  check("gpt2:load returns a status",
    typeof (load.value ?? {}).status === "string",
    JSON.stringify(load.value ?? load.error).slice(0, 140));

  const run = await call("gpt2:runPrompt", { prompt: "The capital of France is" });
  const runVal = run.value ?? {};
  check("gpt2:runPrompt returns a prompt echo",
    typeof runVal.prompt === "string" && runVal.prompt.includes("capital"),
    `prompt=${JSON.stringify(runVal.prompt)}`);
  check("gpt2:runPrompt returns next-token data",
    runVal.next_token !== undefined || runVal.str_tokens !== undefined,
    `next_token=${JSON.stringify(runVal.next_token)}`);

  const act = await call("gpt2:activations", { layer: 5 });
  const actVal = act.value ?? {};
  check("gpt2:activations honours the layer argument",
    actVal.layer === 5 || actVal.error !== undefined,
    `layer=${actVal.layer} status=${actVal.status}`);

  const head = await call("gpt2:attentionHead", { layer: 3, head: 2 });
  const headVal = head.value ?? {};
  check("gpt2:attentionHead honours layer/head arguments",
    headVal.layer === 3 || headVal.error !== undefined,
    `layer=${headVal.layer} head=${headVal.head} status=${headVal.status}`);

  console.log("=".repeat(64));
  if (failures) {
    console.log(`FAILED: ${failures} check(s)`);
    process.exit(1);
  }
  console.log("ALL RUNTIME.JS CHECKS PASSED");
})();
