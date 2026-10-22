# Authentication by build

The control plane requires a credential on every `/api/*` route (see
`backend/core/auth.py`). Where that credential lives depends on which build is
running, and in no build does page JavaScript hold it.

## Desktop (Electron)

```
renderer  --ipc 'api:request'-->  main process  --Authorization: Bearer-->  :8000
```

* `frontend/electron/apiProxy.js` is the only code that reads the token. It
  validates the method, the path (`/api` or `/health` only), the headers and the
  body size, then attaches `Authorization` and returns `{status, body}`.
* `frontend/electron/ipc/api.js` refuses a request whose `senderFrame.url` is
  not our renderer (dev-server origin or `file://` under `dist/`).
* The preload exposes `apiRequest` and nothing else related to credentials. The
  renderer is sandboxed (`sandbox: true`), so the preload cannot reach `fs`.
* `frontend/src/services/api.ts` routes through `apiRequest` whenever that
  bridge exists. `apiToken()` no longer exists, and `authHeaders()` returns only
  the caller's own headers.

Tests: `tests/pytest/test_renderer_token_isolation.py` executes the preload with
a stubbed `electron` module and drives the proxy.

## Browser build

```
browser  --Cookie: mech_session (HttpOnly)-->  :8000
```

A browser has no privileged process, so the token is exchanged once:

```
POST /session            Authorization: Bearer <token>
  -> 200  Set-Cookie: mech_session=<nonce>.<hmac>; HttpOnly; SameSite=Strict; Path=/
DELETE /session          -> clears the cookie
```

* The cookie is signed with the token (`HMAC-SHA256`), so it is stateless and
  **rotating the token invalidates every issued session**.
* `HttpOnly` means page JavaScript cannot read it. `SameSite=Strict` stops a
  *cross-site* request from carrying it — but "same site" is (scheme,
  registrable domain) and **the port is not part of it**
  (RFC 6265bis §5.2.3). A page served from `http://localhost:3000` is
  therefore same-site with `http://localhost:8000`, and the browser *does*
  send the cookie to it. `SameSite=Strict` is not a cross-origin defence;
  the server checks the `Origin` of a cookie-authenticated request itself
  (see below).
* `POST /session` is reachable without a credential — it has to be — but grants
  nothing without a valid token, and returns 401 otherwise.
* `frontend/src/utils/browserPolyfill.js` sends `credentials: 'same-origin'`
  and no `Authorization` header. SSE (`societyService`) relies on the cookie
  too; the token is no longer placed in a URL, where it would land in browser
  history and access logs.

Tests: `tests/pytest/test_web_session_auth.py`.

## Which origins may use the cookie

A cookie-authenticated request must come from an origin the server trusts
(`backend/core/auth.py`). The rule is enforced server-side, in the auth
middleware, on every cookie-authenticated request:

| Origin of the request | Allowed? |
| --- | --- |
| The server's own origin (same host **and** port) — the standalone build serving its own frontend | yes |
| No `Origin` header at all, on a `GET`/`HEAD`/`OPTIONS` | yes (a browser omits `Origin` on a same-origin GET) |
| No `Origin` header at all, on any other method | no |
| An origin listed in `MECH_TRUSTED_ORIGINS` | yes |
| Any loopback origin (`localhost`, `127.0.0.1`, `[::1]`, any port) | only with `MECH_TRUST_LOOPBACK_ORIGINS=1` |
| The opaque origin `null` (a `file://` page, a sandboxed iframe, a `data:` URL) | no |

`MECH_TRUSTED_ORIGINS` is a comma-separated list of exact origins, e.g.
`MECH_TRUSTED_ORIGINS=http://localhost:5173`. `MECH_TRUST_LOOPBACK_ORIGINS=1`
trusts *every* loopback port, which is the cross-port browser dev flow and
nothing else; the default is off, because a loopback port is not an
identity. The opaque `null` origin stays in the CORS *allowlist* (the
packaged desktop app loads from `file://` and needs its responses to be
readable) but it never carries credentials, in CORS or in the session gate.

### Which flow needs which

* **Desktop app** — no cookie at all. `frontend/electron/apiProxy.js` sends
  `Authorization: Bearer` from the main process, so neither the cookie nor
  the origin rules apply to it.
* **Standalone web build** (the backend serves the built frontend) —
  same-origin, so the origin is the server's own and the request is allowed.
* **Browser dev flow** (`npm run dev:renderer`) — the Vite dev server
  proxies `/api` to the backend and attaches `Authorization: Bearer` itself
  (`frontend/vite.config.js`), so the session cookie is not involved. A
  page that talks to the backend *directly* from another port needs
  `MECH_TRUSTED_ORIGINS` (or `MECH_TRUST_LOOPBACK_ORIGINS=1`) plus
  `credentials: 'include'`.

## What is deliberately not here

* No token in `localStorage`, in a Vite env var, or on `window`.
* No query-parameter token by default. `?token=` / `?bearer_token=` is
  refused unless the operator sets `MECH_ALLOW_QUERY_TOKEN=1`: uvicorn's
  access log writes the full query string, so a credential in the URL lands
  in the log in cleartext (and the Electron app pipes backend stdout into
  its logger). The opt-in exists for transports that cannot set a header
  (`EventSource`, `<img src>`); no first-party code emits it.
