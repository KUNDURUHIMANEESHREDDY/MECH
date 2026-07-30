"""Tests for the Python sidecar API dispatcher."""
import pytest
from api.dispatcher import build_dispatcher


def test_build_dispatcher_contains_expected_methods():
    d = build_dispatcher()
    for method in ("ping", "info", "echo", "add", "time", "runtime:status", "runtime:analyze_tokens"):
        assert method in d, f"missing method {method}"


def test_ping_returns_ok():
    d = build_dispatcher()
    result = d["ping"]({})
    assert result["ok"] is True
    assert result["echo"] == "pong"
    assert "timestamp" in result


def test_info_returns_platform_keys():
    d = build_dispatcher()
    result = d["info"]({})
    assert "python" in result
    assert "platform" in result
    assert "pid" in result
    assert isinstance(result["pid"], int)


def test_echo_passes_through():
    d = build_dispatcher()
    assert d["echo"]({"a": 1})["received"] == {"a": 1}


def test_add_sums_numbers():
    d = build_dispatcher()
    assert d["add"]({"a": 2, "b": 40})["sum"] == 42
    assert d["add"]({"a": 0.5, "b": 0.25})["sum"] == 0.75


def test_add_rejects_non_numeric():
    d = build_dispatcher()
    with pytest.raises(ValueError):
        d["add"]({"a": "x", "b": 1})


def test_time_iso_default():
    d = build_dispatcher()
    result = d["time"]({})
    assert result["value"].endswith("Z")


def test_time_unix_format():
    d = build_dispatcher()
    result = d["time"]({"format": "unix"})
    assert isinstance(result["value"], int)


def test_time_human_format():
    d = build_dispatcher()
    result = d["time"]({"format": "human"})
    assert "value" in result and len(result["value"]) == 19


def test_runtime_status():
    d = build_dispatcher()
    result = d["runtime:status"]({})
    assert result["status"] == "ready"


def test_runtime_analyze_tokens():
    d = build_dispatcher()
    result = d["runtime:analyze_tokens"]({"prompt": "hello world"})
    assert result["token_count"] == 2
    assert len(result["tokens"]) == 2
