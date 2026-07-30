from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def send_request(process: subprocess.Popen[str], request: dict[str, object]) -> dict[str, object]:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    assert line
    return json.loads(line)


def test_desktop_service_handles_ping_and_settings(tmp_path: Path) -> None:
    db_path = tmp_path / "desktop.sqlite3"
    process = subprocess.Popen(
        [sys.executable, "-u", "scripts/desktop_service.py", "--db", str(db_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ping = send_request(process, {"id": 1, "method": "ping", "params": {}})
        assert ping["result"]["ok"] is True

        update = send_request(
            process,
            {
                "id": 2,
                "method": "settings.update",
                "params": {"settings": {"theme": "light", "gpuEnabled": True}},
            },
        )
        assert update["result"]["theme"] == "light"
        assert update["result"]["gpuEnabled"] is True

        settings = send_request(
            process,
            {"id": 3, "method": "settings.get", "params": {}},
        )
        assert settings["result"]["theme"] == "light"
    finally:
        process.terminate()
        process.wait(timeout=5)
