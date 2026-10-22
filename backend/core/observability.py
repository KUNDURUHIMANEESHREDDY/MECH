"""Control-plane observability: one configuration object, one logging setup.

Why this module exists
----------------------

The control plane used to read ``os.environ`` at each call site and configure
logging with a single ``logging.basicConfig`` in ``backend/main.py``. That
split the answers to three questions a support engineer asks:

* *What is actually configured?* -- answered differently by every module.
* *Why did this value end up like this?* -- never answered at all; a typo in
  an environment variable silently became a default.
* *What happened during that request?* -- the log lines had no correlation id,
  so the request log and the application log could not be joined.

This module is the single place that answers them:

* :func:`load_settings` reads the environment once, validates every value,
  falls back to the documented default when a value is malformed, and records
  *that it fell back* in ``Settings.problems``.
* :func:`configure_logging` installs one root logging setup: a level taken
  from the settings, a rotating file sink under the storage root, and a
  formatter that redacts credentials.
* Every request carries a correlation id (:func:`new_request_id`) which is
  attached to the log records emitted while the request is handled.

The redaction rule
------------------

Redaction happens inside the formatter, not at the call site. A call site
that logs a token by mistake still cannot leak it: every record passes
through :class:`RedactingFormatter` on its way to a handler. A credential
that is known to the process -- the bearer token -- is replaced wherever it
appears in the formatted line, and anything that *looks* like a credential
(``Authorization: Bearer x``, ``token=y``, ``api_key: z``) is replaced too.

The honest-reporting rule
--------------------------

Nothing here asserts that a subsystem is healthy. This module configures
logging and reports configuration; the probes in ``backend/core/health.py``
decide what is actually working, and an unavailable engine reports the real
reason it is unavailable.
"""

from __future__ import annotations

import datetime as _dt
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: The control-plane version. Also used by ``backend/main.py`` so the
#: version reported by ``GET /`` and by the diagnostics surface cannot drift.
APP_VERSION = "2.0.0"

#: Repository root (the parent of ``backend/``).
REPO_ROOT = Path(__file__).resolve().parents[2]
#: The ``backend/`` package root.
BACKEND_ROOT = REPO_ROOT / "backend"

#: What a redacted value is replaced with.
REDACTED = "***REDACTED***"

#: Log levels the ``log_level`` setting accepts.
LOG_LEVELS: Tuple[str, ...] = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")

#: The log line format. ``request_id`` is filled in by
#: :class:`RedactingFormatter` when it is missing.
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | req=%(request_id)s | %(message)s"


# --------------------------------------------------------------------------- #
# redaction
# --------------------------------------------------------------------------- #

_secret_lock = threading.Lock()
_secrets: set = set()


def register_secret(value: Optional[str]) -> bool:
    """Remember a literal secret so no formatter can emit it.

    Returns ``True`` when the value was registered. Values shorter than four
    characters are refused: replacing a two-character string wherever it
    appears would mangle every message that happens to contain it, and a
    credential that short is not one worth protecting by mangling logs.
    """
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if len(candidate) < 4:
        return False
    with _secret_lock:
        _secrets.add(candidate)
    return True


def _token_file_path() -> Path:
    """The file the bearer token is persisted to.

    Read from ``backend/core/auth.py`` so there is one answer, not two.
    The path is deliberately not configurable -- ``auth.py`` removed that
    seam for good reasons -- so nothing here may reintroduce it.
    """
    try:
        from backend.core import auth

        return Path(auth._token_file())
    except Exception:  # pragma: no cover - auth is always importable
        return BACKEND_ROOT / "storage" / ".mech_api_token"


#: (monotonic timestamp, token) of the last persisted-token lookup. The file
#: is re-read at most every couple of seconds: a log line must not cost a
#: disk read, but the token must not survive a rotation for long.
_token_cache: Tuple[float, Optional[str]] = (0.0, None)
_TOKEN_CACHE_TTL_S = 2.0


def _persisted_token() -> Optional[str]:
    global _token_cache
    now = time.monotonic()
    stamp, cached = _token_cache
    if now - stamp < _TOKEN_CACHE_TTL_S:
        return cached
    value: Optional[str] = None
    try:
        path = _token_file_path()
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            value = text or None
    except OSError:
        value = None
    _token_cache = (now, value)
    return value


def current_bearer_token() -> Optional[str]:
    """The credential this process accepts, if it can be read without making one.

    This never *creates* a token: a log line must not have the side effect of
    writing a credential to disk.
    """
    token = (os.environ.get("MECH_API_TOKEN") or "").strip()
    if token:
        return token
    return _persisted_token()


def _known_secrets() -> List[str]:
    with _secret_lock:
        known = list(_secrets)
    token = current_bearer_token()
    if token:
        known.append(token)
    return known


#: Patterns for credentials that are *shaped* like credentials. Group 1 is
#: kept, the credential itself is replaced.
_SECRET_PATTERNS = (
    (
        re.compile(r"(?i)\b(bearer)([ \t]+)([^\s\"',;]+)"),
        lambda m: "%s%s%s" % (m.group(1), m.group(2), REDACTED),
    ),
    (
        # `authorization: <value>`, but not `authorization: Bearer <value>`:
        # the bearer scheme is handled by the pattern above, so the word
        # "Bearer" is kept and only the credential after it is replaced.
        re.compile(r"(?i)\b(authorization)([ \t]*[:=][ \t]*)(?!bearer\b)([^\s\"',;]+)"),
        lambda m: "%s%s%s" % (m.group(1), m.group(2), REDACTED),
    ),
    (
        re.compile(
            r"(?i)\b([a-z_.-]*(?:token|secret|password|passwd|api[_-]?key)[a-z_.-]*)"
            r"([ \t]*[:=][ \t]*)"
            r"(\"[^\"]*\"|'[^']*'|[^\s,;&\"']+)"
        ),
        lambda m: "%s%s%s" % (m.group(1), m.group(2), REDACTED),
    ),
)


def redact(text: str) -> str:
    """Remove credentials from a string.

    Both the literal secrets this process knows about and anything shaped
    like a credential are replaced. Safe to call on any text.
    """
    if not text:
        return text
    result = text
    for secret in _known_secrets():
        if secret and secret in result:
            result = result.replace(secret, REDACTED)
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class RedactingFormatter(logging.Formatter):
    """A formatter that cannot emit a credential.

    Redaction lives here rather than at the call site on purpose: a call site
    that forgets to sanitise its arguments still cannot leak the token,
    because every record passes through this class on its way to a handler.
    """

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id() or "-"
        return redact(super().format(record))


# --------------------------------------------------------------------------- #
# correlation id
# --------------------------------------------------------------------------- #

_request_id: ContextVar[Optional[str]] = ContextVar("mech_request_id", default=None)


def new_request_id() -> str:
    """A fresh correlation id for one request."""
    return uuid.uuid4().hex


def current_request_id() -> Optional[str]:
    """The correlation id of the request being handled, if any."""
    return _request_id.get()


def bind_request_id(value: Optional[str] = None) -> Tuple[str, Any]:
    """Bind a correlation id to the current context.

    Returns the id and the token :func:`unbind_request_id` needs.
    """
    request_id = value or new_request_id()
    return request_id, _request_id.set(request_id)


def unbind_request_id(token: Any) -> None:
    """Undo a :func:`bind_request_id`."""
    try:
        _request_id.reset(token)
    except ValueError:  # pragma: no cover - reset from another context
        _request_id.set(None)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def _home() -> Path:
    try:
        return Path.home()
    except (RuntimeError, OSError):  # pragma: no cover - no home directory
        return REPO_ROOT


def _default_storage_root() -> str:
    return str(BACKEND_ROOT / "storage")


def _default_export_root() -> str:
    return str(_home() / ".mech" / "exports")


def _default_artifact_dir() -> str:
    return str(BACKEND_ROOT / "storage" / "artifacts")


def _default_model_cache_dir() -> str:
    return str(BACKEND_ROOT / "storage" / "models")


@dataclass
class Settings:
    """The control-plane configuration, with its provenance.

    Every field has a documented default and a documented environment
    variable. ``sources`` records where each value came from
    (``"default"`` or ``"environment:MECH_PORT"``) and ``problems`` records
    every value that was malformed and fell back to its default.
    """

    # -- values ------------------------------------------------------------ #
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    storage_root: str = field(default_factory=_default_storage_root)
    export_root: str = field(default_factory=_default_export_root)
    artifact_dir: str = field(default_factory=_default_artifact_dir)
    model_cache_dir: str = field(default_factory=_default_model_cache_dir)
    engine_timeout_s: float = 120.0
    model_load_timeout_s: float = 600.0
    job_timeout_s: float = 1800.0
    retry_attempts: int = 2
    retry_backoff_s: float = 0.5
    log_file_max_bytes: int = 5 * 1024 * 1024
    log_file_backup_count: int = 3

    # -- provenance -------------------------------------------------------- #
    problems: List[str] = field(default_factory=list)
    sources: Dict[str, str] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------- #
    @property
    def log_dir(self) -> str:
        return str(Path(self.storage_root) / "logs")

    @property
    def log_path(self) -> str:
        return str(Path(self.storage_root) / "logs" / "mech.log")

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-friendly view of the configuration and its provenance."""
        return {
            "log_level": self.log_level,
            "host": self.host,
            "port": self.port,
            "storage_root": self.storage_root,
            "export_root": self.export_root,
            "artifact_dir": self.artifact_dir,
            "model_cache_dir": self.model_cache_dir,
            "engine_timeout_s": self.engine_timeout_s,
            "model_load_timeout_s": self.model_load_timeout_s,
            "job_timeout_s": self.job_timeout_s,
            "retry_attempts": self.retry_attempts,
            "retry_backoff_s": self.retry_backoff_s,
            "log_file_max_bytes": self.log_file_max_bytes,
            "log_file_backup_count": self.log_file_backup_count,
            "log_path": self.log_path,
            "sources": dict(self.sources),
            "problems": list(self.problems),
        }


def _record_str(env: Mapping[str, str], name: str, default: str,
                problems: List[str], sources: Dict[str, str],
                field_name: str) -> str:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        sources[field_name] = "default"
        return default
    value = str(raw).strip()
    sources[field_name] = "environment:%s" % name
    return value


def _record_path(env: Mapping[str, str], name: str, default: str,
                 problems: List[str], sources: Dict[str, str],
                 field_name: str) -> str:
    value = _record_str(env, name, default, problems, sources, field_name)
    if value == default and sources.get(field_name) == "default":
        return value
    try:
        return str(Path(value).expanduser())
    except (TypeError, ValueError):  # pragma: no cover - defensive
        problems.append("%s %r is not a usable path; using %s" % (name, value, default))
        sources[field_name] = "default"
        return default


def _record_int(env: Mapping[str, str], name: str, default: int,
                problems: List[str], sources: Dict[str, str],
                field_name: str, low: int, high: int) -> int:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        sources[field_name] = "default"
        return default
    text = str(raw).strip()
    try:
        value = int(text)
    except ValueError:
        problems.append("%s '%s' is not an integer; using %s" % (name, text, default))
        sources[field_name] = "default"
        return default
    if value < low or value > high:
        problems.append("%s %s is outside %s..%s; using %s"
                        % (name, value, low, high, default))
        sources[field_name] = "default"
        return default
    sources[field_name] = "environment:%s" % name
    return value


def _record_float(env: Mapping[str, str], name: str, default: float,
                  problems: List[str], sources: Dict[str, str],
                  field_name: str, low: float) -> float:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        sources[field_name] = "default"
        return default
    text = str(raw).strip()
    try:
        value = float(text)
    except ValueError:
        problems.append("%s '%s' is not a number; using %s" % (name, text, default))
        sources[field_name] = "default"
        return default
    if value < low:
        problems.append("%s %s is below %s; using %s" % (name, text, low, default))
        sources[field_name] = "default"
        return default
    sources[field_name] = "environment:%s" % name
    return value


def _record_level(env: Mapping[str, str], name: str, default: str,
                  problems: List[str], sources: Dict[str, str],
                  field_name: str) -> str:
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        sources[field_name] = "default"
        return default
    text = str(raw).strip()
    level = text.upper()
    if level not in LOG_LEVELS:
        problems.append("%s '%s' is not a known log level; using %s"
                        % (name, text, default))
        sources[field_name] = "default"
        return default
    sources[field_name] = "environment:%s" % name
    return level


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Read, validate and record the control-plane configuration.

    ``env`` defaults to the process environment. A malformed or out-of-range
    value falls back to its documented default *and* is reported in
    ``Settings.problems`` -- the fallback is never silent.
    """
    source = os.environ if env is None else env
    problems: List[str] = []
    sources: Dict[str, str] = {}

    settings = Settings(
        log_level=_record_level(source, "MECH_LOG_LEVEL", "INFO", problems, sources, "log_level"),
        host=_record_str(source, "MECH_HOST", "127.0.0.1", problems, sources, "host"),
        port=_record_int(source, "MECH_PORT", 8000, problems, sources, "port", 1, 65535),
        storage_root=_record_path(source, "MECH_STORAGE_ROOT",
                                  _default_storage_root(), problems, sources, "storage_root"),
        export_root=_record_path(source, "MECH_EXPORT_ROOT",
                                 _default_export_root(), problems, sources, "export_root"),
        artifact_dir=_record_path(source, "MECH_ARTIFACTS_DIR",
                                  _default_artifact_dir(), problems, sources, "artifact_dir"),
        model_cache_dir=_record_path(source, "MECH_MODEL_CACHE_DIR",
                                     _default_model_cache_dir(), problems, sources, "model_cache_dir"),
        engine_timeout_s=_record_float(source, "MECH_ENGINE_TIMEOUT_S", 120.0,
                                       problems, sources, "engine_timeout_s", 0.001),
        model_load_timeout_s=_record_float(source, "MECH_MODEL_LOAD_TIMEOUT_S", 600.0,
                                           problems, sources, "model_load_timeout_s", 0.001),
        job_timeout_s=_record_float(source, "MECH_JOB_TIMEOUT_S", 1800.0,
                                    problems, sources, "job_timeout_s", 0.001),
        retry_attempts=_record_int(source, "MECH_RETRY_ATTEMPTS", 2,
                                   problems, sources, "retry_attempts", 0, 10),
        retry_backoff_s=_record_float(source, "MECH_RETRY_BACKOFF_S", 0.5,
                                      problems, sources, "retry_backoff_s", 0.0),
        log_file_max_bytes=_record_int(source, "MECH_LOG_MAX_BYTES", 5 * 1024 * 1024,
                                       problems, sources, "log_file_max_bytes", 1024, 1024 * 1024 * 1024),
        log_file_backup_count=_record_int(source, "MECH_LOG_BACKUP_COUNT", 3,
                                          problems, sources, "log_file_backup_count", 0, 100),
    )
    settings.problems = problems
    settings.sources = sources
    return settings


_settings_lock = threading.Lock()
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """The settings in use for this process."""
    global _settings
    with _settings_lock:
        if _settings is None:
            _settings = load_settings()
        return _settings


def reload_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Re-read the configuration and make it the one in use."""
    global _settings
    fresh = load_settings(env)
    with _settings_lock:
        _settings = fresh
    return fresh


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


def configure_logging(settings: Optional[Settings] = None,
                      *, stream: bool = True) -> Dict[str, Any]:
    """Install the control plane's logging setup.

    * the root logger's level comes from the configuration;
    * a rotating file sink is written under the storage root
      (``<storage_root>/logs/mech.log``);
    * every handler uses :class:`RedactingFormatter`, so no credential can
      reach a log line;
    * uvicorn's own handlers are removed so the access log is redacted and
      carries the request id too.

    Returns a summary of what was installed. A file sink that cannot be
    created is reported in the summary and in ``settings.problems`` -- the
    control plane still starts, with console logging only.
    """
    settings = settings or get_settings()
    root = logging.getLogger()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root.setLevel(level)

    # Idempotent: a second call replaces the handlers this function owns
    # instead of stacking a second copy of every line.
    for handler in list(root.handlers):
        if getattr(handler, "_mech_managed", False):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover - defensive
                pass

    formatter = RedactingFormatter(LOG_FORMAT)
    handlers: List[logging.Handler] = []

    log_path = Path(settings.log_path)
    file_enabled = False
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=settings.log_file_max_bytes,
            backupCount=settings.log_file_backup_count,
            encoding="utf-8",
        ))
        file_enabled = True
    except OSError as exc:
        settings.problems.append(
            "log file sink unavailable at %s: %s" % (log_path, exc))

    if stream:
        handlers.append(logging.StreamHandler(sys.stderr))

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        handler._mech_managed = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    token = current_bearer_token()
    if token:
        register_secret(token)

    # The last-error summary the diagnostics surface shows. It is a handler
    # rather than something every call site has to remember to report.
    error_handler = LastErrorHandler()
    error_handler.setLevel(logging.ERROR)
    error_handler._mech_managed = True  # type: ignore[attr-defined]
    root.addHandler(error_handler)

    # uvicorn installs its own handlers with its own format; routing them
    # through ours is what makes the access log redacted and correlatable.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    return {
        "level": settings.log_level,
        "file": str(log_path),
        "file_enabled": file_enabled,
        "handlers": [type(handler).__name__ for handler in handlers],
        "problems": list(settings.problems),
    }


# --------------------------------------------------------------------------- #
# last error
# --------------------------------------------------------------------------- #


@dataclass
class ErrorRecord:
    """One recorded failure."""

    timestamp: str
    kind: str
    message: str
    context: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "message": self.message,
            "context": self.context,
        }


_error_lock = threading.Lock()
_errors: List[ErrorRecord] = []
_MAX_ERRORS = 25


def utcnow_iso() -> str:
    """ISO-8601 UTC, second precision."""
    return (_dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"))


def record_error(error: Any, context: str = "") -> ErrorRecord:
    """Remember a failure so the diagnostics surface can show it.

    The recorded text is redacted: the last error summary is served over the
    API and must not carry a credential either.
    """
    if isinstance(error, BaseException):
        kind = type(error).__name__
        message = str(error)
    else:
        kind = "Error"
        message = str(error)
    record = ErrorRecord(
        timestamp=utcnow_iso(),
        kind=kind,
        message=redact(message),
        context=redact(context),
    )
    with _error_lock:
        _errors.append(record)
        del _errors[:-_MAX_ERRORS]
    return record


def last_error() -> Optional[Dict[str, Any]]:
    """The most recent recorded failure, or ``None`` when there is none."""
    with _error_lock:
        if not _errors:
            return None
        return _errors[-1].as_dict()


def recent_errors(limit: int = 10) -> List[Dict[str, Any]]:
    """The most recent recorded failures, newest last."""
    try:
        count = max(1, int(limit))
    except (TypeError, ValueError):
        count = 10
    with _error_lock:
        return [record.as_dict() for record in _errors[-count:]]


class LastErrorHandler(logging.Handler):
    """Records ERROR and CRITICAL log records as the last error.

    This is what makes the diagnostics surface useful without every call
    site having to remember to report its own failures.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= logging.ERROR:
                record_error("%s: %s" % (record.name, record.getMessage()),
                             context="log:%s" % record.levelname)
        except Exception:  # pragma: no cover - a handler must never raise
            pass
