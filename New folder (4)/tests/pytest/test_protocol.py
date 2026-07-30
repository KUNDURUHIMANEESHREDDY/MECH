"""End-to-end test of the Python sidecar by spawning the real ``main.py``
process and speaking the JSON-lines protocol over stdio.
"""
import json
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PY_MAIN = os.path.abspath(os.path.join(HERE, "..", "..", "backend", "main.py"))


def _spawn():
    return subprocess.Popen(
        [sys.executable, PY_MAIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def _read_line(proc, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            return line.rstrip("\n")
        time.sleep(0.05)
    raise TimeoutError("sidecar did not respond")


def _request(proc, method, payload=None, msg_id=1):
    msg = json.dumps({"id": msg_id, "method": method, "payload": payload or {}})
    proc.stdin.write(msg + "\n")
    proc.stdin.flush()
    deadline = time.time() + 5
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("id") == msg_id:
            return obj
    raise TimeoutError(f"no response for {method}")


@pytest.fixture
def sidecar():
    proc = _spawn()
    try:
        # wait for the "ready" banner
        ready = _read_line(proc)
        assert json.loads(ready).get("type") == "ready"
        yield proc
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_ping_roundtrip(sidecar):
    res = _request(sidecar, "ping", msg_id=10)
    assert res["result"]["ok"] is True


def test_add_roundtrip(sidecar):
    res = _request(sidecar, "add", {"a": 7, "b": 35}, msg_id=11)
    assert res["result"]["sum"] == 42


def test_unknown_method_returns_error(sidecar):
    res = _request(sidecar, "does_not_exist", {}, msg_id=12)
    assert "error" in res
    assert "does_not_exist" in res["error"]


def test_echo_roundtrip(sidecar):
    res = _request(sidecar, "echo", {"hello": "world"}, msg_id=13)
    assert res["result"]["received"] == {"hello": "world"}
