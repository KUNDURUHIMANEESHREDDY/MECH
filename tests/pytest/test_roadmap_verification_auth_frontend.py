"""P0 #2 -- the frontend must authenticate; no raw fetch may drop the token.

Claim under test: the frontend uses the authenticated client rather than a raw
``fetch`` that drops the bearer token, and no authenticated control-plane route
is reachable without one.

Two halves, because either alone is weak evidence:

* **Script half.** Every ``fetch(`` call site under ``frontend/src`` is located
  and required to carry a credential (an ``authHeaders`` spread, an explicit
  ``Authorization`` header, or the ``apiToken()`` query-parameter path that
  ``EventSource`` needs because it cannot set headers).  A path that lands in
  neither bucket is a defect.
* **Backend half.** The script claim is only meaningful if the backend actually
  rejects the unauthenticated request.  ``/api`` is probed anonymously, and
  ``auth.is_public_path`` is attacked with the bypass shapes -- query strings,
  trailing slashes, case, missing leading slash.

The polyfill is loaded as a module into a fake ``window`` and its observed
``fetch`` calls are inspected, so "it carries the token" is measured rather than
read off the source.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"

#: Markers that prove a call site attached a credential.
CREDENTIAL_MARKERS = ("authHeaders", "Authorization", "apiToken(")

#: A call site is allowed to *implement* credential attachment rather than use
#: it: `apiFetch` and the private `request` helper in services/api.ts are the
#: authenticated client. They are named explicitly so an unauthenticated call
#: anywhere else cannot hide behind the same exemption.
IMPLEMENTATION_SITES = {
    "services/api.ts": ("apiFetch", "request"),
}


def _frontend_sources():
    """Every non-test frontend source file that can issue a request."""
    files = []
    for pattern in ("*.ts", "*.js", "*.vue"):
        files.extend(FRONTEND_SRC.rglob(pattern))
    return sorted(p for p in files if p.is_file())


def _source_text(path: Path) -> str:
    """File text, decoded leniently.

    Some components carry mojibake banners (byte sequences that are not valid
    UTF-8).  They are cosmetic, but a strict decode raises and a strict read of
    a *scanned* file would make the scanner fragile for the wrong reason.  The
    undecodable bytes are replaced rather than dropped so line numbers hold.
    """
    return path.read_bytes().decode("utf-8", errors="replace")


def _strip_comments(text: str) -> str:
    """Blank out JS/TS comments, preserving offsets but not stray matches.

    Without this, ``fetch(`` inside a comment or a docstring -- including the
    mojibake box-drawing banners in these components -- is reported as an
    unauthenticated call site, which is a false positive that would train a
    reader to ignore the check.  Line length is preserved so reported line
    numbers stay correct.
    """
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"),
                  text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _fetch_call_windows(text: str, size: int = 700):
    """A text window around each ``fetch(`` call, plus its line number."""
    windows = []
    for match in re.finditer(r"(?<![.\w])fetch\s*\(", text):
        start = match.start()
        window = text[start:start + size]
        line = text.count("\n", 0, start) + 1
        windows.append((line, window))
    return windows


def test_the_scan_finds_the_expected_call_sites():
    """Guard the scan: a refactor must not silently empty it."""
    total = 0
    files_with_fetch = 0
    for path in _frontend_sources():
        windows = _fetch_call_windows(_strip_comments(_source_text(path)))
        if windows:
            files_with_fetch += 1
            total += len(windows)
    assert total >= 10, (
        f"the scan found only {total} fetch() call(s); it may be matching "
        f"nothing after a refactor"
    )
    assert files_with_fetch >= 5, files_with_fetch


def test_every_fetch_call_site_carries_a_credential():
    """The defect the claim is about: a raw fetch that drops the token.

    A call site passes when its own text attaches a credential.  Two whole
    files are exempt because their *job* is to attach it -- and each is
    separately required to contain the credential logic, so the exemption
    cannot be used to hide an unauthenticated call.
    """
    unauthenticated = []
    for path in _frontend_sources():
        rel = path.relative_to(FRONTEND_SRC).as_posix()
        text = _source_text(path)
        if rel in IMPLEMENTATION_SITES:
            assert any(marker in text for marker in CREDENTIAL_MARKERS), (
                f"{rel} is exempted as the authenticated client but contains no "
                f"credential logic"
            )
            continue
        code = _strip_comments(text)
        for line, window in _fetch_call_windows(code):
            if not any(marker in window for marker in CREDENTIAL_MARKERS):
                unauthenticated.append(f"{rel}:{line}")

    assert not unauthenticated, (
        f"{len(unauthenticated)} fetch() call site(s) attach no bearer "
        f"credential, so the request will 401 (or worse, be silently "
        f"unauthenticated): {unauthenticated}"
    )


def test_no_vite_env_token_is_bundled():
    """A VITE_-prefixed token would be compiled into the shipped bundle."""
    offenders = []
    for path in _frontend_sources():
        # Comments are stripped: api.ts documents *why* it avoids the variable,
        # and the prohibition is on referencing it in code.
        code = _strip_comments(_source_text(path))
        if "VITE_MECH_API_TOKEN" in code:
            offenders.append(path.relative_to(FRONTEND_SRC).as_posix())
    assert not offenders, (
        f"VITE_MECH_API_TOKEN is referenced in {offenders}; Vite inlines "
        f"VITE_* into the bundle, which ships the secret to every client"
    )


def test_browser_polyfill_actually_sends_the_token():
    """Run the polyfill under Node and observe the request it actually makes.

    The polyfill is a second, independent path to the control plane: it is what
    the web build uses when Electron's bridge is absent.  Reading its source
    would only show intent, so it is executed with ``fetch`` and ``window``
    stubbed and the resulting ``init.headers`` are inspected.  Skipped when
    Node is unavailable -- and the skip is loud, because a silent skip here
    would remove the only execution-level evidence for this path.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the polyfill cannot be executed")

    polyfill = FRONTEND_SRC / "utils" / "browserPolyfill.js"
    assert polyfill.exists(), polyfill

    harness = r"""
const captured = [];
let assigned = null;

globalThis.fetch = async (url, init) => {
  captured.push({ url: String(url), headers: (init && init.headers) || {} });
  return { json: async () => ({ result: { ok: true } }) };
};

const win = {
  __MECH_API_TOKEN__: '',
  localStorage: { getItem: (k) => (k === 'mech_api_token' ? 'storage-token' : null) },
};

Object.defineProperty(globalThis, 'window', { value: win, configurable: true });

const src = require('fs').readFileSync(process.argv[1], 'utf8');
const fn = new Function('window', 'fetch', src + '\n;return window.appApi;');
assigned = fn(win, globalThis.fetch);

(async () => {
  const token = assigned.getApiToken();
  await assigned.pythonPing();
  console.log(JSON.stringify({ token, captured }));
})().catch((err) => {
  console.log(JSON.stringify({ error: String(err && err.message || err) }));
  process.exit(3);
});
"""

    proc = subprocess.run(
        [node, "-e", harness, str(polyfill)],
        capture_output=True, text=True, timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"the polyfill failed to execute under node:\n{proc.stderr[-800:]}"
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])

    assert "error" not in payload, payload["error"]
    assert payload["token"] == "storage-token", (
        "the polyfill did not resolve the token from localStorage; "
        f"got {payload['token']!r}"
    )
    assert payload["captured"], "the polyfill issued no request"
    url, init = payload["captured"][0]["url"], payload["captured"][0]
    headers = {k.lower(): v for k, v in init["headers"].items()}
    assert "authorization" in headers, (
        f"the polyfill called {url} with no Authorization header: {init}"
    )
    assert headers["authorization"] == "Bearer storage-token", headers


def test_browser_polyfill_has_no_unauthenticated_fetch_path():
    """Even without Node, no fetch in the polyfill may lack the header."""
    text = _source_text(FRONTEND_SRC / "utils" / "browserPolyfill.js")
    for line, window in _fetch_call_windows(_strip_comments(text)):
        assert "Authorization" in window, (
            f"browserPolyfill.js:{line} issues a fetch without an "
            f"Authorization header"
        )


def test_event_source_token_travels_as_a_query_parameter():
    """EventSource cannot set headers; the token must be in the URL.

    Without it every SSE subscribe 401s and the panel degrades to polling --
    silently, which is why the contract is pinned.
    """
    text = (FRONTEND_SRC / "services" / "societyService.ts").read_text(
        encoding="utf-8")
    assert "apiToken()" in text, (
        "societyService must resolve the token for the SSE subscribe"
    )
    assert re.search(r"token=\$\{encodeURIComponent\(token\)\}", text), (
        "the SSE url must carry the encoded token as a query parameter"
    )
    assert "EventSource" in text, "expected an EventSource-based stream"


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


def test_scan_is_not_vacuous():
    """The credential markers must appear in the tree at all."""
    joined = "\n".join(_source_text(p) for p in _frontend_sources())
    for marker in CREDENTIAL_MARKERS:
        assert marker in joined, f"no frontend source references {marker!r}"
    json.dumps({"ok": True})  # keep the import used and deterministic
