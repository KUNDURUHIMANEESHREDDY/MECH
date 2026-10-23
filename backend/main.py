"""MECH Platform - FastAPI Server Entry Point.

This is the backend entry used by the Electron app in both development and the
packaged .exe. Keep /api routes available without needing a separate server.
"""
import asyncio
import importlib.util
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Mount

# When this file is executed directly (`python backend/main.py`) Python puts
# `backend/` on sys.path, not the repository root, so the string import in
# `uvicorn.run("backend.main:app")` below would fail with
# `ModuleNotFoundError: No module named 'backend'`. Put the repo root first.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Imported after the path fix above. The credential policy -- which origins
# may use the session cookie, which may carry credentials cross-origin --
# lives in one place: backend/core/auth.py.
from backend.core import auth as auth_mod
from backend.core.recovery import (
    EngineFailure,
    EngineTimeout,
    call_with_timeout,
    retry_call,
)

# One logging setup for the control plane. The level comes from the
# configuration, a rotating file sink is written under the storage root, and
# the formatter redacts credentials -- so no call site can leak the bearer
# token by logging it. This replaces the ad-hoc `logging.basicConfig` call
# that used to live here. See backend/core/observability.py.
from backend.core import observability as observability_mod

_settings = observability_mod.get_settings()
_LOGGING_SUMMARY = observability_mod.configure_logging(_settings)
logger = logging.getLogger("MECH")

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application startup/shutdown.

    Replaces the deprecated `@app.on_event` hooks. Teardown checkpoints the
    SQLite WAL so a clean shutdown does not leave a large -wal file behind.
    A checkpoint failure is logged and swallowed: it is not a reason to fail
    shutdown, and the WAL remains durable either way.
    """
    logger.info("MECH starting up (lifespan).")
    try:
        from backend.core import auth as auth_mod
        # The token is registered as a secret, so even if some future call
        # site logs it, the formatter removes it from the line.
        observability_mod.register_secret(auth_mod.get_bearer_token())
        logger.info("Control-plane bearer auth active (token file: %s).",
                    auth_mod._token_file())
    except Exception as exc:
        logger.warning("Bearer token init failed: %s", exc)
    try:
        from backend.core import evidence_graph as eg_mod
        report = eg_mod.quarantine_legacy_records()
        if report.get("moved"):
            logger.info("Quarantined %s legacy evidence record(s) to %s.",
                        report["moved"], report["destination"])
    except Exception as exc:
        logger.warning("Evidence quarantine skipped: %s", exc)
    try:
        from backend.plugins.service import get_service
        restored = get_service().restore_enabled()
        if restored:
            logger.info("Restored enabled plugins: %s", ", ".join(restored))
    except Exception as exc:
        logger.warning("Plugin restore skipped: %s", exc)

    await _preload_gpt2_engine()

    # The MCP HTTP transport only serves requests from inside its session
    # manager's task group, and a mounted sub-application never receives a
    # lifespan of its own, so the task group is started here.
    #
    # The mount itself is NOT installed here: it is registered at import
    # time (see `_install_mcp_mount`), so the route table is the same
    # before the application has started and after it has started a
    # hundred times. Startup swaps the transport behind the mount.
    global _mcp_session_cm
    _mcp_session_cm = None
    try:
        transport = _build_mcp_transport()
        from backend.mcp_server.server import mcp as _mech_mcp
        _mcp_session_cm = _mech_mcp.session_manager.run()
        await _mcp_session_cm.__aenter__()
        _MCP_HOLDER.set_transport(transport)
        logger.info("MCP control plane mounted at /mcp (tools: mech_*).")
    except Exception as exc:
        _mcp_session_cm = None
        _MCP_HOLDER.set_transport(None)
        logger.warning("MCP control plane failed to start; /mcp is absent: %s", exc)

    try:
        yield
    finally:
        logger.info("MECH shutting down (lifespan).")
        try:
            from backend.plugins.service import get_service
            get_service().shutdown()
        except Exception as exc:
            logger.warning("Plugin worker shutdown incomplete: %s", exc)
        if _mcp_session_cm is not None:
            try:
                await _mcp_session_cm.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("MCP session manager shutdown failed: %s", exc)
        # The transport goes with the task group that served it. Leaving it
        # in place would answer requests from a session manager that is no
        # longer running.
        _MCP_HOLDER.set_transport(None)
        try:
            from backend.storage.database import checkpoint_wal
            busy, log_pages, checkpointed = checkpoint_wal()
            if busy:
                logger.warning(
                    "WAL checkpoint busy at shutdown (busy=%s log=%s "
                    "checkpointed=%s); the WAL stays durable but is not "
                    "truncated.", busy, log_pages, checkpointed)
            else:
                logger.info("SQLite WAL checkpoint complete at shutdown "
                            "(%s pages).", log_pages)
        except Exception as exc:
            logger.warning("WAL checkpoint failed at shutdown: %s", exc)


app = FastAPI(
    title="MECH Research Platform",
    version=observability_mod.APP_VERSION,
    description="Mechanistic Interpretability Research Platform",
    lifespan=lifespan,
)


def _cors_origins() -> list[str]:
    """Exact origins permitted to call the API.

    Electron loads from file://, which browsers report as the opaque origin
    "null"; it stays allowed because the packaged desktop app depends on it.
    Loopback origins on arbitrary ports are handled by
    ``_LOOPBACK_ORIGIN_REGEX`` instead, since Vite binds a random port when
    5173 is taken. MECH_CORS_ORIGINS may add exact origins; entries are
    normalized so a trailing slash cannot silently fail to match, and an
    entry that is not a well-formed origin is refused rather than allowed
    by accident.
    """
    origins = ["null"]
    for extra in os.environ.get("MECH_CORS_ORIGINS", "").split(","):
        extra = extra.strip()
        if not extra:
            continue
        normalized = auth_mod.normalize_origin(extra)
        if normalized is None:
            logger.warning(
                "MECH_CORS_ORIGINS: ignoring %r -- not a valid origin "
                "(scheme://host[:port] and nothing else).", extra)
            continue
        if normalized not in origins:
            origins.append(normalized)
    return origins


# Loopback only, on any port, http or https. The trailing group is anchored
# so that "http://localhost.evil.com" and "http://127.0.0.1.attacker.org" do
# NOT match -- a subdomain must not inherit loopback trust.
_LOOPBACK_ORIGIN_REGEX = auth_mod.LOOPBACK_ORIGIN_REGEX


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=_LOOPBACK_ORIGIN_REGEX,
    # Credentials are added by _cors_credentials_middleware, and only for the
    # origins that are allowed to carry them. allow_credentials=True here
    # would hand them to every origin in the allowlist, including a remote
    # host the operator added to MECH_CORS_ORIGINS.
    allow_credentials=False,
    # Only the verbs and headers this API actually uses.
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    # So a browser client can read the correlation id out of the response.
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def _cors_credentials_middleware(request: Request, call_next):
    """Add Access-Control-Allow-Credentials only where it is safe.

    The CORS middleware runs with allow_credentials=False so that no origin
    in the allowlist gets credentialed access by accident. This puts the
    header back for the origins that are allowed to use the operator's
    credentials: loopback origins, and the origins the operator listed in
    MECH_TRUSTED_ORIGINS. The opaque "null" origin stays in the CORS
    allowlist (the packaged desktop app loads from file:// and needs its
    responses to be readable) but never carries credentials, exactly like
    the session gate, which refuses it.
    """
    origin = request.headers.get("origin")
    response = await call_next(request)
    if origin and auth_mod.origin_may_carry_credentials(origin):
        response.headers["access-control-allow-credentials"] = "true"
    return response


async def _preload_gpt2_engine() -> None:
    """Pre-load GPT-2 so the first interactive request does not time out.

    Failure here is not fatal: the platform stays up and every dependent view
    degrades to seeded/unavailable responses with an explicit reason. The
    reason is recorded, so the diagnostics surface shows why the weights are
    not in memory instead of leaving the user to guess.

    The load is bounded by the configured model-load timeout and retried a
    bounded number of times. Retrying is safe here: a load is idempotent, a
    failed load leaves no weights behind, and the previous attempt has
    finished before the next one starts. A timeout is never retried -- the
    thread that overran may still be loading.
    """
    logger.info("Backend ready. ML model modules will load on demand.")
    try:
        from backend.services import gpt2_engine
        if gpt2_engine.is_available():
            logger.info("Pre-loading GPT-2 engine (torch/transformers)...")
            settings = observability_mod.get_settings()

            def _load_once():
                outcome = call_with_timeout(
                    gpt2_engine.load,
                    settings.model_load_timeout_s,
                    label="GPT-2 model load",
                )
                if not isinstance(outcome, dict) or outcome.get("status") != "loaded":
                    reason = ((outcome or {}).get("error")
                              or (outcome or {}).get("status")
                              or "the engine did not report a loaded model")
                    raise EngineFailure(reason)
                return outcome

            result, attempts = await asyncio.to_thread(
                retry_call,
                _load_once,
                max(1, settings.retry_attempts),
                settings.retry_backoff_s,
            )
            logger.info("GPT-2 engine pre-loaded: status=%s (attempt %s)",
                        result.get("status", "unknown"), attempts)
        else:
            logger.info("GPT-2 engine not available - using seeded fallbacks.")
    except EngineTimeout as exc:
        logger.warning("GPT-2 pre-loading timed out: %s", exc)
        observability_mod.record_error(exc, context="GPT-2 preload")
    except Exception as exc:
        logger.warning("GPT-2 pre-loading failed: %s", exc)
        observability_mod.record_error(exc, context="GPT-2 preload")


@app.middleware("http")
async def _correlation_id_middleware(request: Request, call_next):
    """Give every request a correlation id, and put it in the response.

    The id travels in the ``X-Request-ID`` response header and in every log
    line emitted while the request is handled, so an access-log line and the
    application log can be joined without guessing from timestamps.

    An id supplied by the caller is honoured, because it is what makes a
    report from a user's browser traceable -- but it is sanitised first: an
    id that could contain a newline would let a caller forge log lines.
    """
    request_id, token = observability_mod.bind_request_id(
        _incoming_request_id(request.headers.get("x-request-id")))
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info("%s %s -> %s (request %s)",
                    request.method, request.url.path, response.status_code,
                    request_id)
        return response
    except Exception:
        observability_mod.record_error(
            sys.exc_info()[1],
            context="unhandled error in %s %s" % (request.method, request.url.path),
        )
        raise
    finally:
        observability_mod.unbind_request_id(token)


def _incoming_request_id(raw: Optional[str]) -> Optional[str]:
    """Accept a caller-supplied correlation id only when it is safe to print.

    Anything that is not a short run of letters, digits, dashes and
    underscores is refused: a correlation id is written into every log line
    of a request, so an id that carries a newline would let the caller
    write log lines of their own.
    """
    if not raw:
        return None
    candidate = str(raw).strip()
    if not candidate or len(candidate) > 64:
        return None
    if not all(char.isalnum() or char in "-_" for char in candidate):
        return None
    return candidate


@app.middleware("http")
async def _request_log_middleware(request: Request, call_next):
    """Record request metadata for the Logging view (no bodies stored)."""
    started = time.perf_counter()
    response = await call_next(request)
    try:
        if request.url.path != "/health":
            from backend.core.request_log import record
            record(
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - started) * 1000.0,
                request_id=observability_mod.current_request_id() or "",
            )
    except Exception:
        pass
    return response


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """Mandatory bearer gate for the control plane.

    Loopback reachability and Origin are never authorization. ``/`` and
    ``/health`` stay public, CORS preflight (OPTIONS) passes through to the
    CORS middleware, and the built frontend's static files stay public so the
    app can load. Everything else under ``/api``, ``/mcp``, ``/health/``,
    ``/openapi.json``, ``/docs`` and ``/redoc`` requires
    ``Authorization: Bearer <MECH_API_TOKEN>`` or a session cookie minted
    from it by ``POST /session``.

    A cookie-authenticated request is additionally checked against the
    Origin the browser sent: the cookie is ambient (the browser attaches
    it to any same-site request, and same-site ignores the port), so the
    server has to verify the origin itself instead of trusting that the
    cookie arrived. See ``backend/core/auth.py``.
    """
    if auth_mod.is_public_path(request.url.path, request.method):
        return await call_next(request)
    token = auth_mod.extract_bearer(request.headers.get("authorization"))
    if token is None:
        # ?token= is refused unless the operator opted in: the query string
        # is written to the access log, so the credential would be readable
        # there in cleartext.
        token = auth_mod.extract_query_token(request.query_params)

    if token is not None:
        if not auth_mod.verify_bearer(token):
            return JSONResponse(
                status_code=401,
                content={"detail": "invalid credentials"},
            )
        return await call_next(request)

    # No bearer credential: a browser session cookie is the other accepted form.
    # It is HttpOnly, so page JavaScript cannot read it even when it is present.
    session = auth_mod.extract_session_cookie(request.headers.get("cookie"))
    if session is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "missing credentials"},
        )
    if not auth_mod.verify_session(session):
        return JSONResponse(
            status_code=401,
            content={"detail": "invalid credentials"},
        )
    # The cookie travels with every same-site request, and same-site ignores
    # the port: a page on any loopback port would otherwise present the
    # operator's cookie. The Origin has to be one this server trusts.
    if not auth_mod.origin_allowed_for_session(
            request.headers.get("origin"),
            request.headers.get("host") or request.url.netloc,
            request.method):
        return JSONResponse(
            status_code=403,
            content={"detail": "untrusted origin"},
        )
    return await call_next(request)


def _body_cap() -> int:
    """Global request-body ceiling in bytes (MECH_MAX_BODY_BYTES override).

    Prompts have their own 2048-char bound and SAE corpora 200k chars, but
    the rest of the control plane accepts untyped `Dict[str, Any]` bodies.
    Without a ceiling an authenticated caller can push arbitrary megabytes at
    the dispatcher, so bound at the edge and let endpoint schemas be tighter.
    """
    raw = os.environ.get("MECH_MAX_BODY_BYTES", "").strip()
    try:
        value = int(raw) if raw else 2 * 1024 * 1024
    except ValueError:
        value = 2 * 1024 * 1024
    return max(1024, min(value, 64 * 1024 * 1024))


async def _read_limited_body(request: Request, cap: int) -> bytes:
    """Read request body up to `cap + 1` bytes, enforcing the limit.

    Always counts actual bytes streamed, regardless of Content-Length header.
    Returns the complete body (buffered) so downstream can re-read it.
    """
    total = 0
    chunks = []
    async for chunk in request.stream():
        total += len(chunk)
        if total > cap:
            raise ValueError(f"request body exceeds {cap} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


@app.middleware("http")
async def _body_size_middleware(request: Request, call_next):
    """Reject oversized bodies before they reach the application.

    Always counts actual bytes streamed, regardless of Content-Length header.
    A lying or chunked header cannot slip a large payload past.
    Buffers the body so downstream (FastAPI, Pydantic) can re-read it.
    """
    cap = _body_cap()
    content_length = request.headers.get("content-length")

    # Fast path: declared length exceeds cap -> reject immediately
    if content_length is not None:
        try:
            if int(content_length) > cap:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"request body exceeds {cap} bytes"},
                )
        except ValueError:
            return JSONResponse(
                status_code=400, content={"detail": "malformed content-length"},
            )

    # Read and count actual bytes, buffering for downstream
    try:
        body = await _read_limited_body(request, cap)
    except ValueError as exc:
        return JSONResponse(status_code=413, content={"detail": str(exc)})

    # Replace request._body so downstream (FastAPI, Pydantic) can read it
    request._body = body

    # Also set content-length so downstream sees consistent header
    request.headers.__dict__["_list"] = [
        (k, v) for k, v in request.headers.raw if k != b"content-length"
    ] + [(b"content-length", str(len(body)).encode())]

    return await call_next(request)


#: The middleware stack, OUTERMOST first. Starlette's ``add_middleware``
#: inserts at position 0, so the LAST registration is the OUTERMOST
#: middleware -- the effective order is the opposite of the reading order,
#: and it changes whenever a middleware is added. This list is the order
#: the security model needs, and the tests assert it:
#:
#:   1. correlation id  - every request gets an id, and every response and
#:                        log line carries it.
#:   2. CORS credentials - decides who may carry credentials cross-origin.
#:   3. CORS             - every response (401/403/413 included) carries CORS
#:                         headers, so a browser client can read the refusal.
#:   4. auth gate        - rejects an unauthenticated request BEFORE its body
#:                         is read, so an anonymous caller cannot make the
#:                         server buffer megabytes.
#:   5. body ceiling     - bounds the body of an authenticated request.
#:   6. request log      - records what happened.
_MIDDLEWARE_ORDER = (
    "_correlation_id_middleware",
    "_cors_credentials_middleware",
    "CORSMiddleware",
    "_auth_middleware",
    "_body_size_middleware",
    "_request_log_middleware",
)


def _middleware_name(entry) -> str:
    """The name of a middleware in the stack (its dispatch function or class)."""
    dispatch = entry.kwargs.get("dispatch")
    if dispatch is not None:
        return getattr(dispatch, "__name__", "")
    return getattr(entry.cls, "__name__", "")


def _enforce_middleware_order() -> None:
    """Put the middleware stack in the order the security model requires.

    Registration order is an accident waiting to happen: adding a
    middleware above would silently move the auth gate behind the body
    ceiling again, and the symptom (an anonymous 3 MB POST answered with
    413 instead of 401) is invisible in a green suite. The order is
    therefore stated once, here, and asserted by the tests.
    """
    priority = {name: index for index, name in enumerate(_MIDDLEWARE_ORDER)}
    app.user_middleware[:] = sorted(
        app.user_middleware,
        key=lambda entry: priority.get(_middleware_name(entry), len(priority)),
    )


_enforce_middleware_order()


@app.get("/")
def root():
    return {"name": "MECH Platform", "version": observability_mod.APP_VERSION,
            "status": "running"}


@app.post("/session")
def create_session(request: Request):
    """Exchange the bearer token for an HttpOnly session cookie.

    A browser has no privileged process to hold the token, so the token is
    presented once (by the operator, or by the desktop shell) and replaced by a
    cookie JavaScript cannot read. The cookie is signed with the token, so
    rotating the token invalidates every issued session.

    This route is reachable without a credential -- it has to be, or nothing
    could ever obtain one -- but it grants nothing without a valid token.
    """
    token = auth_mod.extract_bearer(request.headers.get("authorization"))
    if token is None:
        # Same opt-in as the gate: ?token= lands in the access log.
        token = auth_mod.extract_query_token(request.query_params)
    if token is None or not auth_mod.verify_bearer(token):
        return JSONResponse(
            status_code=401,
            content={"detail": "invalid credentials"},
        )

    response = JSONResponse({"status": "ok"})
    response.set_cookie(
        auth_mod.SESSION_COOKIE,
        auth_mod.issue_session(),
        max_age=12 * 60 * 60,
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
        path="/",
    )
    return response


@app.delete("/session")
def delete_session():
    """Clear the session cookie. Idempotent and requires no credential."""
    from backend.core import auth as auth_mod

    response = JSONResponse({"status": "ok"})
    response.delete_cookie(auth_mod.SESSION_COOKIE, path="/")
    return response


@app.get("/health")
def health():
    """Liveness only. Deliberately does not touch ML or storage subsystems.

    A liveness probe that loads torch or writes to disk turns a slow dependency
    into an orchestrator restart loop. For what the deployment can actually do,
    see /health/subsystems.
    """
    return {"status": "healthy"}


@app.get("/health/subsystems")
def health_subsystems(subsystem: Optional[str] = None):
    """Prove each subsystem rather than assuming it.

    Reports process, dependency, model, executor, storage, and evidence health.
    The status vocabulary has no 'healthy' member: nothing here proves health,
    only that a subsystem was probed and what it reported.
    """
    from backend.core.health import health_snapshot, run_probes, overall

    if subsystem:
        wanted = [s.strip() for s in subsystem.split(",") if s.strip()]
        results = run_probes(wanted)
        return {"status": overall(results), "subsystems": results}

    return health_snapshot()


# Main frontend services call /api/*. Keep legacy /api/v1/* too for older views.
# The dispatcher is the core router: if it fails to load, every frontend feature
# is missing and the platform is useless. Fail loudly at startup rather than
# serving a half-working app that merely *looks* healthy.
try:
    from backend.api.dispatcher import router as api_router

    app.include_router(api_router, prefix="/api")
    app.include_router(api_router, prefix="/api/v1")
    logger.info("API dispatcher loaded at /api and /api/v1.")
except Exception:
    logger.exception("API dispatcher failed to load; the platform cannot serve "
                     "/api or /api/v1. Refusing to start.")
    raise

# Optional v2 runtime API. `backend.api.runtime_api` exists today and is mounted
# at /api/v1 and /api/v2, but the find_spec guard stays for trees that lack it:
# absence is an expected condition and gets a warning, while a *present but
# broken* module is an infrastructure failure and is logged with a traceback.
# Either way /api/v2 is never absent silently.
try:
    if importlib.util.find_spec("backend.api.runtime_api") is None:
        logger.warning(
            "/api/v2 not mounted: backend.api.runtime_api does not exist. "
            "Any client calling /api/v2 will get a 404 -- capability absent, "
            "not degraded."
        )
    else:
        from backend.api.runtime_api import router as runtime_router

        # One router, two prefixes. The spec documents jobs at /api/v1/jobs while
        # /api/v2 is where they were first exposed; mounting the single
        # implementation again is safer than copying it, because there is still
        # exactly one QUEUE behind both paths.
        app.include_router(runtime_router, prefix="/api/v1")
        app.include_router(runtime_router, prefix="/api/v2")
        logger.info("Runtime v2 API loaded at /api/v1 and /api/v2.")
except Exception:
    logger.exception("Runtime v2 API module exists but failed to load; "
                     "/api/v2 is absent.")


# Contract routers (spec 21.1 endpoint families, 21.2 event stream). Each lives
# in its own module and each failure is isolated: a broken contract router must
# not remove the dispatcher the whole frontend depends on. Absence is a logged
# capability gap; a present-but-broken module is an infrastructure failure and
# is logged with a traceback. Neither is ever mounted silently.
for _module_name, _label in (
    ("backend.api.v1_contracts", "v1 contract API (21.1)"),
    ("backend.api.events", "job event stream (21.2)"),
    ("backend.api.export_contracts", "project export archive (19.3)"),
    ("backend.api.diagnostics", "diagnostics and configuration API"),
):
    try:
        if importlib.util.find_spec(_module_name) is None:
            logger.warning(
                "%s not mounted: %s does not exist. A client calling it gets a "
                "404 -- capability absent, not degraded.", _label, _module_name,
            )
            continue
        _contract_module = importlib.import_module(_module_name)
        # Dual mount, same convention as the dispatcher: the frontend calls
        # /api/*, the spec documents /api/v1/*, and both resolve to one router.
        app.include_router(_contract_module.router, prefix="/api")
        app.include_router(_contract_module.router, prefix="/api/v1")
        logger.info("%s loaded at /api and /api/v1 from %s.",
                    _label, _module_name)
    except Exception:
        logger.exception("%s exists but failed to load; it is absent from "
                         "/api and /api/v1.", _label)


# MCP control plane: same Core API the UI calls, exposed as tools so external
# agents drive MECH without UI automation. Optional: a failed mount must not
# take the REST API or the UI down with it.
#
# The transport is mounted by the lifespan, not at import time: FastMCP's
# streamable HTTP app only serves requests from inside its session manager's
# task group, and a mounted sub-application never receives a lifespan of its
# own. Mounted at import time it answered every authenticated request with
# "RuntimeError: Task group is not initialized".
_mcp_mounts: list = []
_mcp_session_cm = None


def _mcp_allowed_hosts() -> list[str]:
    """Host header values the MCP HTTP transport answers on.

    The MCP SDK's DNS-rebinding protection is on by default and refuses
    every Host that is not listed, so the transport has to be told which
    hosts it is reachable on. Loopback is the default bind (MECH_BIND_HOST
    defaults to 127.0.0.1); a wider bind is a deliberate operator choice
    and is allowed as well. Origins are not allowed: the MCP clients are
    agents, not pages, and the transport sits behind the bearer gate.
    """
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    bind_host = os.environ.get("MECH_BIND_HOST", "").strip()
    if bind_host and bind_host not in ("127.0.0.1", "localhost", "::1"):
        hosts.append(f"{bind_host}:*")
    return hosts


def _build_mcp_transport():
    """Build the MCP HTTP transport with a fresh session manager."""
    from backend.mcp_server.server import mcp as _mech_mcp
    from mcp.server.transport_security import TransportSecuritySettings
    # FastMCP caches the session manager on the FastMCP instance and that
    # manager refuses to run() twice, so the cached one is dropped first: a
    # second startup in this process (a test, a reload) needs its own.
    _mech_mcp._session_manager = None
    _mech_mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_mcp_allowed_hosts(),
    )
    return _mech_mcp.streamable_http_app()


class _McpTransportHolder:
    """The mounted ``/mcp`` endpoint, holding whichever transport is live.

    The mount exists from import time, and the transport behind it is
    swapped at startup. Two reasons, and the second is the important one.

    *The transport has to be rebuilt at each startup.* The MCP session
    manager refuses to run() twice, so a second startup in this process (a
    test, a reload) needs its own manager and therefore its own transport.

    *The mount must not depend on whether the application has started.* The
    route table is read by the OpenAPI schema, by the route-authorization
    matrix, and by anyone introspecting the app. A route that appears only
    after startup makes all of those depend on startup history: the same
    process would publish a different API before and after its first
    request. That is not a difference in the API, it is a difference in
    when the API was asked.

    So the mount points at this holder, the holder points at the current
    transport, and swapping the transport never touches the route table.
    """

    def __init__(self) -> None:
        self.transport: Optional[object] = None

    def set_transport(self, transport: Optional[object]) -> None:
        self.transport = transport

    async def __call__(self, scope, receive, send) -> None:
        transport = self.transport
        if transport is None:
            # Not started, or shut down. The endpoint exists in the route
            # table, so it has to answer -- and what it says is that the
            # control plane is not running, not a 404 that would suggest
            # the path does not exist.
            response = JSONResponse(
                {"detail": "the MCP control plane is not running"},
                status_code=503,
            )
            await response(scope, receive, send)
            return
        await transport(scope, receive, send)


#: The mounted ``/mcp`` endpoint. Populated at startup; the mount itself is
#: registered at import time so the route table does not depend on it.
_MCP_HOLDER = _McpTransportHolder()


def _install_mcp_mount(transport) -> None:
    """Mount the MCP transport at /mcp, in front of the catch-all frontend.

    The built frontend is mounted at "/" and matches every path, so the MCP
    mount has to come before it: mounted behind it, the static handler
    answers first and every /mcp request is a 405.

    The mount is installed ONCE, at import time, and points at
    ``_MCP_HOLDER``. Startup swaps the transport behind the holder rather
    than re-mounting, so the route table is the same before the application
    has ever run and after it has run ten times.
    """
    global _mcp_mounts
    if _mcp_mounts:
        return
    mount = Mount("/mcp", app=_MCP_HOLDER)
    routes = app.router.routes
    index = len(routes)
    for position, route in enumerate(routes):
        if getattr(route, "path", "") == "/":
            index = position
            break
    routes.insert(index, mount)
    _mcp_mounts = [mount]


def _find_frontend_dist() -> Optional[str]:
    """Locate the built Vue frontend (frontend/dist) for standalone mode.

    Standalone = one process serves both UI and API, so the browser can just
    open http://127.0.0.1:8000 with no Electron and no second server.
    Returns the dist directory path, or None when not built / not bundled.
    """
    candidates = [
        os.path.join(_REPO_ROOT, "frontend", "dist"),  # repo checkout
        os.path.join(os.getcwd(), "frontend", "dist"),  # launched from root
        os.path.join(os.getcwd(), "dist"),  # portable MECH-standalone/dist
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dist"),
        os.path.join(getattr(sys, "_MECH_RESOURCES", ""), "app", "dist"),
    ]
    for cand in candidates:
        if not cand:
            continue
        norm = os.path.normpath(cand)
        index_html = os.path.join(norm, "index.html")
        if os.path.isdir(norm) and os.path.isfile(index_html):
            return norm
    return None


try:
    _DIST_DIR = _find_frontend_dist()
    if _DIST_DIR is not None:
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="frontend")
        logger.info("Serving standalone frontend from %s at /", _DIST_DIR)
    else:
        logger.info(
            "No frontend dist found; running API-only. "
            "Build it with: cd frontend && npm run build:renderer"
        )
except Exception:
    logger.exception("Frontend static mount failed; running API-only.")

# The MCP endpoint is mounted at import time, after the static mount, so
# that it sits in front of it (the static handler matches every path) and
# so that the route table is identical before and after startup. The
# transport behind the mount is installed by the lifespan.
_install_mcp_mount(None)


if __name__ == "__main__":
    logger.info("Starting MECH Platform backend on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False, timeout_keep_alive=600)
