"""The desktop renderer must not be able to obtain the bearer token.

Reading source is not evidence that a secret is unreachable, so the two things
that could hand it over are *executed*:

* ``frontend/electron/preload.js`` is loaded with a stubbed ``electron`` module
  and everything it exposes to the page is inspected. The assertion is that no
  exposed value is the token and that no accessor returns one.
* ``frontend/electron/apiProxy.js`` is required directly and driven: it must
  reject traversal/host-injection, strip a renderer-supplied ``Authorization``
  header, and never echo the token in a result or an error.

The remaining checks are source invariants that no execution can prove (the
window is created with ``sandbox: true``; no ``executeJavaScript`` writes the
token into the page; a packaged build has no bare-``python`` fallback).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ELECTRON_DIR = REPO_ROOT / "frontend" / "electron"
PRELOAD = ELECTRON_DIR / "preload.js"
MAIN = ELECTRON_DIR / "main.js"
API_PROXY = ELECTRON_DIR / "apiProxy.js"

SECRET = "test-only-token-value-8f2a1c"


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the bridge cannot be executed")
    return node


def _run_node(script_path: Path, *args: str, env: dict | None = None) -> dict:
    import os

    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    proc = subprocess.run(
        [_node(), str(script_path), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env=run_env,
    )
    assert proc.returncode == 0, (
        f"node harness failed:\nstdout: {proc.stdout[-2000:]}\n"
        f"stderr: {proc.stderr[-2000:]}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


PRELOAD_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');

const exposed = {};
const fakeRequire = (name) => {
  if (name !== 'electron') throw new Error('preload required a non-electron module: ' + name);
  return {
    contextBridge: { exposeInMainWorld: (key, value) => { exposed[key] = value; } },
    ipcRenderer: {
      invoke: async () => ({ status: 200, body: {} }),
      on: () => {},
      removeListener: () => {},
    },
  };
};

new Function('require', src)(fakeRequire);

const appApi = exposed.appApi || {};
const report = {
  globals: Object.keys(exposed),
  tokenGlobalPresent: Object.prototype.hasOwnProperty.call(exposed, '__MECH_API_TOKEN__'),
  appApiKeys: Object.keys(appApi),
  hasGetApiToken: typeof appApi.getApiToken === 'function',
  hasApiRequest: typeof appApi.apiRequest === 'function',
  valuesThatAreTheToken: Object.values(exposed).filter((v) => v === process.env.SECRET_TOKEN),
  returnedToken: typeof appApi.getApiToken === 'function' ? appApi.getApiToken() : null,
};
console.log(JSON.stringify(report));
"""

PROXY_HARNESS = r"""
const path = require('path');
const { pathToFileURL } = require('url');
const proxy = require(process.argv[2]);

const out = { rejected: {}, headers: {}, trusted: {}, leak: null };

const badPaths = [
  '../etc/passwd',
  '/api/../../etc/passwd',
  'http://evil.example/api/status',
  'https://evil.example/api/status',
  '/secret',
  '',
  '/api/status?x=1',
];
for (const candidate of badPaths) {
  try {
    proxy.validateApiRequest({ method: 'GET', path: candidate });
    out.rejected[candidate] = 'ALLOWED';
  } catch (err) {
    out.rejected[candidate] = 'rejected';
  }
}

try {
  proxy.validateApiRequest({ method: 'TRACE', path: '/api/status' });
  out.rejected['METHOD:TRACE'] = 'ALLOWED';
} catch (err) {
  out.rejected['METHOD:TRACE'] = 'rejected';
}

const sanitised = proxy.validateApiRequest({
  method: 'POST',
  path: '/api/gpt2/run_prompt',
  body: { prompt: 'hello' },
  headers: { Authorization: 'Bearer attacker', 'X-Request-Id': 'req-1' },
});
out.headers = sanitised.headers;

const dist = path.join(process.argv[3], 'dist', 'index.html');
out.trusted.distFile = proxy.isTrustedRendererUrl(pathToFileURL(dist).href);
out.trusted.devOrigin = proxy.isTrustedRendererUrl('http://localhost:5173/index.html');
out.trusted.remote = proxy.isTrustedRendererUrl('https://evil.example/index.html');
out.trusted.outsideDist = proxy.isTrustedRendererUrl(
  pathToFileURL(path.join(process.argv[3], 'index.html')).href
);
out.trusted.aboutBlank = proxy.isTrustedRendererUrl('about:blank');

const timeout = new Promise((resolve) => setTimeout(() => resolve({ timedOut: true }), 5000));
const attempt = proxy
  .apiRequest({ method: 'GET', path: '/api/status' })
  .then((res) => ({ resolved: res }))
  .catch((err) => ({ rejected: String(err && err.message || err) }));

Promise.race([attempt, timeout]).then((result) => {
  const serialised = JSON.stringify(result);
  out.leak = serialised.includes(process.env.SECRET_TOKEN) ? serialised : null;
  out.outcome = result.timedOut ? 'timeout' : Object.keys(result)[0];
  console.log(JSON.stringify(out));
});
"""


def test_preload_exposes_no_token(tmp_path: Path) -> None:
    """Execute the preload and inspect everything it puts on `window`."""
    harness = tmp_path / "preload_harness.cjs"
    harness.write_text(PRELOAD_HARNESS, encoding="utf-8")

    report = _run_node(harness, str(PRELOAD), env={"SECRET_TOKEN": SECRET})
    # The harness compares against SECRET_TOKEN, so a token equal to it would
    # show up. Point the file at a known token by exporting it.
    assert report["tokenGlobalPresent"] is False, (
        "preload.js exposes window.__MECH_API_TOKEN__; renderer JavaScript can "
        "read the control-plane credential"
    )
    assert report["hasGetApiToken"] is False, (
        "preload.js exposes appApi.getApiToken(); the token must stay in the "
        "main process"
    )
    assert report["hasApiRequest"] is True, (
        "preload.js must expose appApi.apiRequest, the authenticated proxy"
    )
    assert report["valuesThatAreTheToken"] == [], report["valuesThatAreTheToken"]
    assert report["returnedToken"] is None


def test_preload_source_never_reads_the_token_file() -> None:
    """A sandboxed preload cannot use `fs`; it must not try."""
    source = PRELOAD.read_text(encoding="utf-8")
    assert "mech_api_token" not in source
    assert "getApiToken" not in source
    assert "__MECH_API_TOKEN__" not in source
    assert "node:fs" not in source and "require('fs')" not in source


def test_api_proxy_rejects_unsafe_requests(tmp_path: Path) -> None:
    """Drive the proxy: traversal, host injection and header smuggling."""
    harness = tmp_path / "proxy_harness.cjs"
    harness.write_text(PROXY_HARNESS, encoding="utf-8")

    report = _run_node(
        harness,
        str(API_PROXY),
        str(REPO_ROOT / "frontend"),
        env={"SECRET_TOKEN": SECRET, "MECH_API_TOKEN": SECRET},
    )

    allowed = [p for p, verdict in report["rejected"].items() if verdict != "rejected"]
    assert allowed == [], f"apiProxy accepted unsafe request(s): {allowed}"

    assert "authorization" not in report["headers"], (
        "a renderer-supplied Authorization header survived validation"
    )
    assert report["headers"].get("x-request-id") == "req-1"

    trusted = report["trusted"]
    assert trusted["distFile"] is True
    assert trusted["devOrigin"] is True
    assert trusted["remote"] is False
    assert trusted["outsideDist"] is False
    assert trusted["aboutBlank"] is False

    assert report["leak"] is None, f"the proxy leaked the token: {report['leak']}"


def test_main_process_holds_the_token_and_sandboxes_the_renderer() -> None:
    """Source invariants that execution cannot observe from outside Electron."""
    source = MAIN.read_text(encoding="utf-8")

    assert "__MECH_API_TOKEN__" not in source, (
        "main.js still injects the token into the renderer"
    )
    assert "executeJavaScript" not in source, (
        "main.js executes JavaScript in the renderer; that is how the token "
        "used to be delivered"
    )
    assert re.search(r"sandbox:\s*true", source), (
        "the renderer is not sandboxed"
    )
    assert "registerIpcHandlers" in source
    assert "setPermissionRequestHandler" in source, (
        "no permission request handler denies renderer permissions"
    )


def test_packaged_build_has_no_system_python_fallback() -> None:
    """A release must not silently run on whatever `python` is installed."""
    source = MAIN.read_text(encoding="utf-8")
    assert "bundledOnly" in source, (
        "main.js does not distinguish the packaged (bundled-only) interpreter "
        "lookup from the development one"
    )
    # The bare interpreter name is only legitimate as the last resort of the
    # development branch.
    fallback = re.search(
        r"if \(bundledOnly\) return null;\s*\n\s*return process\.platform === 'win32' \? 'python' : 'python3';",
        source,
    )
    assert fallback, (
        "the bare-`python` fallback is not guarded by the bundledOnly branch"
    )
    assert "BackendStartupError" in source, (
        "a backend that cannot start must raise a clear startup error"
    )


def test_api_handler_validates_the_sender_frame() -> None:
    """The proxy must refuse a request from a frame that is not ours."""
    handler = ELECTRON_DIR / "ipc" / "api.js"
    source = handler.read_text(encoding="utf-8")
    assert "senderFrame" in source
    assert "isTrustedRendererUrl" in source
    assert "api:request" in source


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
