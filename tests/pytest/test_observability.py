"""Tests for the control plane's logging, configuration and diagnostics.

Three properties are proved here, and each one is a property the platform
depends on:

* the bearer token never reaches a log line;
* a malformed configuration value falls back to its documented default and
  says that it did;
* a failed engine call is reported as a failure, never as a success.
"""

import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core import health as health_mod  # noqa: E402
from backend.core import observability as obs  # noqa: E402


@pytest.fixture()
def restore_logging():
    """Put the logging setup back the way it was.

    ``configure_logging`` owns the root logger. Without this the file sink
    and the root level installed by one test would leak into the next.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_uvicorn = {
        name: (list(logging.getLogger(name).handlers),
               logging.getLogger(name).propagate)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
    }
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if handler not in saved_handlers:
                root.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        for name, (handlers, propagate) in saved_uvicorn.items():
            logger = logging.getLogger(name)
            logger.handlers[:] = handlers
            logger.propagate = propagate


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


def test_the_bearer_token_never_reaches_a_log_line(tmp_path, monkeypatch,
                                                   restore_logging):
    """A credential logged by mistake is removed by the formatter.

    The redaction lives in the formatter, not at the call site, so this
    test logs the token the way a careless call site would and checks the
    line that reached the file.
    """
    token = "super-secret-token-value-123456"
    monkeypatch.setenv("MECH_API_TOKEN", token)

    settings = obs.load_settings({"MECH_STORAGE_ROOT": str(tmp_path)})
    summary = obs.configure_logging(settings, stream=False)
    assert summary["file_enabled"] is True

    logger = logging.getLogger("MECH.test.token")
    logger.info("calling the API with Authorization: Bearer %s", token)
    logger.warning("the raw credential %s was in this message", token)

    for handler in logging.getLogger().handlers:
        handler.flush()

    text = Path(summary["file"]).read_text(encoding="utf-8")
    assert token not in text, "the bearer token reached the log file"
    assert obs.REDACTED in text, "the credential was not marked as redacted"
    assert "calling the API with Authorization: Bearer" in text


def test_a_credential_shaped_value_is_redacted_even_when_it_is_not_the_token(
        tmp_path, monkeypatch, restore_logging):
    """Anything that looks like a credential is redacted.

    The live token is one secret; a call site can hold another one (an
    upstream key, a session id). The formatter cannot know which is which,
    so it removes both.
    """
    monkeypatch.setenv("MECH_API_TOKEN", "the-real-token-value-1234")
    settings = obs.load_settings({"MECH_STORAGE_ROOT": str(tmp_path)})
    summary = obs.configure_logging(settings, stream=False)

    logger = logging.getLogger("MECH.test.shape")
    logger.info("api_key=sk-should-never-appear-in-a-log")

    for handler in logging.getLogger().handlers:
        handler.flush()

    text = Path(summary["file"]).read_text(encoding="utf-8")
    assert "sk-should-never-appear-in-a-log" not in text


def test_a_request_gets_a_correlation_id_in_the_response_and_in_the_log(
        tmp_path, monkeypatch, restore_logging):
    """The id is in the response header and in the request's log lines."""
    from fastapi.testclient import TestClient

    from backend.core import auth as auth_mod
    from backend.main import app

    monkeypatch.setenv("MECH_API_TOKEN", "correlation-token-value")
    auth_mod.reset_cache()

    settings = obs.load_settings({"MECH_STORAGE_ROOT": str(tmp_path)})
    summary = obs.configure_logging(settings, stream=False)

    # No context manager: the lifespan pre-loads GPT-2, which downloads
    # weights. These tests are about the control plane, not about the model.
    client = TestClient(app)
    try:
        response = client.get("/api/status",
                              headers={"Authorization": "Bearer correlation-token-value"})
        assert response.status_code == 200
        request_id = response.headers.get("X-Request-ID")
        assert request_id, "the response carried no correlation id"
        assert len(request_id) >= 8
    finally:
        auth_mod.reset_cache()

    for handler in logging.getLogger().handlers:
        handler.flush()
    text = Path(summary["file"]).read_text(encoding="utf-8")
    assert "req=%s" % request_id in text, "the log line did not carry the id"


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def test_a_malformed_setting_falls_back_to_the_default_and_reports_it():
    """A bad value never silently becomes a default."""
    settings = obs.load_settings({
        "MECH_PORT": "abc",
        "MECH_LOG_LEVEL": "LOUD",
        "MECH_ENGINE_TIMEOUT_S": "-5",
        "MECH_HOST": "",
    })

    assert settings.port == 8000
    assert settings.log_level == "INFO"
    assert settings.engine_timeout_s == 120.0
    assert settings.host == "127.0.0.1"

    problems = "\n".join(settings.problems)
    assert "MECH_PORT" in problems and "abc" in problems
    assert "MECH_LOG_LEVEL" in problems
    assert "MECH_ENGINE_TIMEOUT_S" in problems

    # The provenance says where the value came from: a fallback is recorded
    # as a fallback, not as a configured value.
    assert settings.sources["port"] == "default"
    assert settings.sources["log_level"] == "default"
    assert settings.sources["engine_timeout_s"] == "default"


def test_a_valid_setting_is_used_and_its_source_is_recorded():
    """A good value is used, and the environment is named as its source."""
    settings = obs.load_settings({
        "MECH_PORT": "9123",
        "MECH_LOG_LEVEL": "debug",
        "MECH_ENGINE_TIMEOUT_S": "45",
    })

    assert settings.port == 9123
    assert settings.log_level == "DEBUG"
    assert settings.engine_timeout_s == 45.0
    assert settings.problems == []
    assert settings.sources["port"] == "environment:MECH_PORT"
    assert settings.sources["log_level"] == "environment:MECH_LOG_LEVEL"


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #


def test_the_diagnostics_route_is_behind_the_auth_gate(monkeypatch):
    """No credential, no diagnostics."""
    from fastapi.testclient import TestClient

    from backend.core import auth as auth_mod
    from backend.main import app

    monkeypatch.setenv("MECH_API_TOKEN", "diagnostics-token-value")
    auth_mod.reset_cache()

    client = TestClient(app)
    try:
        assert client.get("/api/diagnostics").status_code == 401
        assert client.get("/api/config").status_code == 401

        headers = {"Authorization": "Bearer diagnostics-token-value"}
        response = client.get("/api/diagnostics", headers=headers)
        assert response.status_code == 200
        payload = response.json()

        # The subsystems are probed, not asserted: every one of them reports
        # a status from the probe vocabulary, and none of them says
        # "healthy".
        assert payload["subsystems"], "no subsystem was probed"
        for name, probe in payload["subsystems"].items():
            assert probe.get("status") in (
                health_mod.OK, health_mod.DEGRADED,
                health_mod.UNAVAILABLE, health_mod.UNKNOWN,
            ), "%s reported an invented status" % name
            assert "healthy" not in probe.get("status", "")

        assert payload["app_version"]
        assert payload["configuration"]["port"]
        assert isinstance(payload["configuration_problems"], list)
        assert "storage" in payload
        assert "disk" in payload["storage"]
        assert "dependencies" in payload
        assert "last_error" in payload

        config = client.get("/api/config", headers=headers)
        assert config.status_code == 200
        assert "configuration" in config.json()
    finally:
        auth_mod.reset_cache()


def test_a_failed_engine_call_is_reported_as_a_failure_not_a_success(monkeypatch):
    """A broken engine must not be reported as a working one."""
    from backend.api import diagnostics as diag
    from backend.services import gpt2_engine

    # The ML stack is missing entirely.
    monkeypatch.setattr(gpt2_engine, "is_available", lambda: False)
    report = diag._engine_report()
    assert report["status"] != health_mod.OK
    assert report["status"] == health_mod.UNAVAILABLE
    assert report["weights_loaded"] is False
    assert report["reason"], "an unavailable engine must give the reason"

    # The stack is present but the engine itself fails.
    def _boom():
        raise RuntimeError("weights are corrupt")

    monkeypatch.setattr(gpt2_engine, "is_available", lambda: True)
    monkeypatch.setattr(gpt2_engine, "info", _boom)
    report = diag._engine_report()
    assert report["status"] != health_mod.OK
    assert report["weights_loaded"] is False
    assert "weights are corrupt" in report["reason"]


def test_the_diagnostics_surface_does_not_claim_a_broken_engine_is_loaded(
        monkeypatch):
    """The API payload carries the failure, not a success."""
    from fastapi.testclient import TestClient

    from backend.core import auth as auth_mod
    from backend.main import app
    from backend.services import gpt2_engine

    monkeypatch.setenv("MECH_API_TOKEN", "diagnostics-token-value")
    monkeypatch.setattr(gpt2_engine, "is_available", lambda: False)
    auth_mod.reset_cache()

    client = TestClient(app)
    try:
        response = client.get(
            "/api/diagnostics",
            headers={"Authorization": "Bearer diagnostics-token-value"},
        )
        assert response.status_code == 200
        payload = response.json()
        model = payload["engines"]["model"]
        assert model["status"] != health_mod.OK
        assert model["weights_loaded"] is False
        assert model["reason"]
    finally:
        auth_mod.reset_cache()
