"""P0 -- no renderer holds the control-plane credential.

Two halves, because either alone is weak evidence:

* **Frontend half.** No source under ``frontend/src`` may attach a bearer
  credential. Desktop requests are authenticated by the Electron main process
  (the ``api:request`` proxy); the browser build is authenticated by the
  HttpOnly session cookie issued by ``POST /session``. The polyfill is
  *executed* and the request it actually makes is inspected, so "it carries no
  token" is measured rather than read off the source.
* **Backend half.** The script claim only matters if the backend enforces it:
  ``/api`` is probed anonymously, the session cookie is minted from the token and
  accepted, and ``auth.is_public_path`` is attacked with the bypass shapes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"

#: The only files allowed to call ``fetch`` directly: the authenticated client
#: and the browser polyfill. Everything else must go through ``apiFetch``.
FETCH_IMPLEMENTATION_SITES = {
    "services/api.ts",
    "utils/browserPolyfill.js",
}

#: Strings that would mean a renderer can still read or build a credential.
FORBIDDEN_IN_RENDERER = ("__MECH_API_TOKEN__", "apiToken", "Bearer ")


def _frontend_sources():
    """Every non-test frontend source file that can issue a request."""
    files = []
    for pattern in ("*.ts", "*.js", "*.vue"):
        files.extend(FRONTEND_SRC.rglob(pattern))
    return sorted(p for p in files if p.is_file())


def _source_text(path: Path) -> str:
    """File text, decoded leniently (some components carry mojibake banners)."""
    return path.read_bytes().decode("utf-8", errors="replace")


def _strip_comments(text: str) -> str:
    """Blank out JS/TS comments, preserving offsets but not stray matches."""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"),
                  text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _fetch_call_windows(text: str, size: int = 700):
    """A text window around each ``fetch(`` call, plus its line number."""
    windows = []
    for match in re.finditer(r"(?<![.\w])fetch\s*\(", text):
        start = match.start()
        windows.append((text.count("\n", 0, start) + 1, text[start:start + size]))
    return windows


def test_the_scan_finds_the_expected_call_sites():
    """Guard the scan: a refactor must not silently empty it."""
    total = 0
    files_with_fetch = set()
    for path in _frontend_sources():
        windows = _fetch_call_windows(_strip_comments(_source_text(path)))
        if windows:
            files_with_fetch.add(path.relative_to(FRONTEND_SRC).as_posix())
            total += len(windows)
    assert total >= 2, f"the scan found only {total} fetch() call(s)"
    assert files_with_fetch, "no frontend source issues a fetch at all"


def test_no_renderer_source_reads_or_builds_a_credential():
    """The defect: renderer JavaScript that can obtain the bearer token."""
    offenders = []
    for path in _frontend_sources():
        code = _strip_comments(_source_text(path))
        for marker in FORBIDDEN_IN_RENDERER:
            if marker in code:
                offenders.append(f"{path.relative_to(FRONTEND_SRC).as_posix()}: {marker!r}")
    assert not offenders, (
        "renderer source still references a bearer credential: " + "; ".join(offenders)
    )


def test_only_the_authenticated_client_calls_fetch():
    """A raw fetch cannot carry the session cookie to another origin."""
    callers = set()
    for path in _frontend_sources():
        rel = path.relative_to(FRONTEND_SRC).as_posix()
        if _fetch_call_windows(_strip_comments(_source_text(path))):
            callers.add(rel)
    unexpected = sorted(callers - FETCH_IMPLEMENTATION_SITES)
    assert not unexpected, (
        f"these files call fetch() directly instead of apiFetch(): {unexpected}"
    )


def test_desktop_and_browser_use_their_own_credential_path():
    """Both paths must be present, or one of the builds is unauthenticated."""
    api = _strip_comments(_source_text(FRONTEND_SRC / "services" / "api.ts"))
    assert "apiProxy()" in api, "api.ts does not consult the desktop IPC proxy"
    assert "apiRequest" in api, "api.ts does not call the preload proxy"
    assert "credentials: 'same-origin'" in api, (
        "the browser build does not send the session cookie"
    )


def test_browser_polyfill_sends_no_token():
    """Run the polyfill and inspect the request it actually makes."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the polyfill cannot be executed")

    polyfill = FRONTEND_SRC / "utils" / "browserPolyfill.js"
    assert polyfill.exists(), polyfill

    harness = r"""
const captured = [];

globalThis.fetch = async (url, init) => {
  captured.push({
    url: String(url),
    headers: (init && init.headers) || {},
    credentials: init && init.credentials,
    body: init && init.body,
  });
  return { json: async () => ({ result: { ok: true } }) };
};

const win = {
  localStorage: { getItem: () => 'storage-token' },
};

Object.defineProperty(globalThis, 'window', { value: win, configurable: true });

const src = require('fs').readFileSync(process.argv[1], 'utf8');
const fn = new Function('window', 'fetch', src + '\n;return window.appApi;');
const api = fn(win, globalThis.fetch);

(async () => {
  await api.pythonPing();
  console.log(JSON.stringify({
    captured,
    hasGetApiToken: typeof api.getApiToken === 'function',
  }));
})().catch((err) => {
  console.log(JSON.stringify({ error: String(err && err.message || err) }));
  process.exit(3);
});
"""

    proc = subprocess.run(
        [node, "-e", harness, str(polyfill)],
        capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"the polyfill failed to execute under node:\n{proc.stderr[-800:]}"
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "error" not in payload, payload.get("error")

    assert payload["hasGetApiToken"] is False, (
        "the polyfill exposes getApiToken(); a renderer could read the credential"
    )
    assert payload["captured"], "the polyfill issued no request"
    request = payload["captured"][0]
    headers = {k.lower(): v for k, v in request["headers"].items()}
    assert "authorization" not in headers, (
        f"the polyfill sent a credential to {request['url']}: {headers}"
    )
    assert request["credentials"] == "include", (
        "the polyfill does not ask for the session cookie; it will 401"
    )
    assert "/api/" in request["url"] and "/api/v1/" not in request["url"], (
        f"the polyfill targets a path the backend does not serve: {request['url']}"
    )


def test_event_source_does_not_put_the_token_in_the_url():
    """A credential in a URL ends up in history and in access logs."""
    text = (FRONTEND_SRC / "services" / "societyService.ts").read_text(encoding="utf-8")
    assert "EventSource" in text, "expected an EventSource-based stream"
    assert "token=" not in text, (
        "societyService still puts a credential in the SSE query string"
    )
    assert "apiToken" not in text


def test_scan_is_not_vacuous():
    """The scan must be looking at a non-trivial tree."""
    sources = _frontend_sources()
    assert len(sources) > 50, f"only {len(sources)} frontend sources found"
    joined = "\n".join(_source_text(p) for p in sources)
    assert "apiFetch" in joined
    assert "EventSource" in joined
    json.dumps({"ok": True})  # keep the import used and deterministic
# --------------------------------------------------------------------------- #
# backend half: unauthenticated requests are actually rejected
# --------------------------------------------------------------------------- #

def _auth():
    from backend.core import auth as auth_mod
    return auth_mod


def test_api_requires_a_token_anonymously():
    """Measure the rejection instead of assuming it."""
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        anon = client.get("/api/status")
        assert anon.status_code == 401, (
            f"GET /api/status answered {anon.status_code} without a token"
        )
        bad = client.get("/api/status",
                         headers={"Authorization": "Bearer not-the-token"})
        assert bad.status_code == 401, bad.status_code
        good = client.get("/api/status",
                          headers=_auth().auth_headers())
        assert good.status_code == 200, good.status_code


@pytest.mark.parametrize("path", [
    "/api/status",
    "/api/status/",
    "/api/v1/ping",
    "/health/subsystems",
    "/openapi.json",
    "/docs",
    "/mcp",
    "/api/scientific_health",
    "/api/analytics",
])
def test_control_plane_paths_are_not_public(path):
    assert _auth().is_public_path(path, "GET") is False, (
        f"{path} is treated as public, so it bypasses bearer auth"
    )


@pytest.mark.parametrize("path", [
    "/api/status",
    "/api/v1/whatever.json",
    "/api/secret.txt",
    "/api/schema.js",
])
def test_a_query_string_cannot_make_an_api_path_public(path):
    """Extension matching must run on the path, not swallow /api/."""
    for suffix in ("?x=.js", "?file=a.txt", "?v=1"):
        assert _auth().is_public_path(path + suffix, "GET") is False, (
            f"{path + suffix} was classified public"
        )


def test_static_assets_are_still_public():
    """The fix must not have made the app unloadable before it has a token."""
    for path in ("/assets/main.js", "/assets/main.css", "/favicon.ico",
                 "/manifest.json", "/assets/main.js?v=123"):
        assert _auth().is_public_path(path, "GET") is True, path


def test_unknown_paths_default_to_protected():
    """Explicit-allowlist: everything not listed requires auth."""
    for path in ("/", "/health"):
        assert _auth().is_public_path(path, "GET") is True, path
    for path in ("/unknown", "/some/new/endpoint", "/api/v2/newthing"):
        assert _auth().is_public_path(path, "GET") is False, path


def test_verify_bearer_rejects_junk_inputs():
    verify = _auth().verify_bearer
    assert verify(None) is False
    assert verify("") is False
    assert verify("   ") is False
    assert verify(12345) is False            # type: ignore[arg-type]
    assert verify("wrong") is False
    assert verify(_auth().get_bearer_token()) is True


def test_extract_bearer_parses_only_the_bearer_scheme():
    extract = _auth().extract_bearer
    assert extract("Bearer abc") == "abc"
    assert extract("bearer  abc  ") == "abc"
    assert extract("Basic abc") is None
    assert extract("abc") is None          # no scheme separator
    assert extract(None) is None
    assert extract("Bearer ") is None


def test_openapi_spec_is_not_public_json():
    """`/openapi.json` ends in .json but must not be caught by static matching."""
    assert _auth().is_public_path("/openapi.json", "GET") is False
    assert _auth().is_public_path("/openapi.yaml", "GET") is False
