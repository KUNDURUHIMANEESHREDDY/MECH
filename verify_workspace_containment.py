"""Live check of workspace.describe containment through the real sidecar.

Drives frontend/scripts/desktop_service.py over its actual JSON-lines IPC
protocol, the way electron/electron/ipc/pythonBridge.ts does. This is the
reachable surface: the renderer's workspace.describe maps straight here.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SERVICE = REPO_ROOT / "frontend" / "scripts" / "desktop_service.py"

errors = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def call_many(requests, env_extra, db):
    env = dict(os.environ)
    env.update(env_extra or {})
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    proc = subprocess.run(
        [sys.executable, str(SERVICE), "--db", str(db)],
        input=payload, capture_output=True, text=True,
        timeout=180, env=env, cwd=str(REPO_ROOT),
    )
    out = []
    for line in proc.stdout.strip().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    if not out:
        return [{"error": {"message": proc.stderr[-300:]}}]
    return out


def ok(reply):
    return isinstance(reply, dict) and "result" in reply and "error" not in reply


def err(reply):
    return isinstance(reply, dict) and isinstance(reply.get("error"), dict)


def message(reply):
    return (reply or {}).get("error", {}).get("message", "")


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    allowed = tmp / "work"
    (allowed / "sub").mkdir(parents=True)
    (allowed / "a.py").write_text("x", encoding="utf-8")
    (allowed / "sub" / "b.py").write_text("x", encoding="utf-8")

    outside = tmp / "outside"
    outside.mkdir()
    for i in range(5):
        (outside / f"secret{i}.txt").write_text("s", encoding="utf-8")

    db = tmp / "mech.db"
    roots = {"MECH_WORKSPACE_ROOTS": str(allowed)}

    print("=" * 62)
    print("workspace.describe containment — live process")
    print("=" * 62)

    results = call_many([
        {"method": "workspace.describe", "params": {"path": str(allowed)}},
        {"method": "workspace.describe", "params": {"path": str(outside)}},
        {"method": "workspace.describe", "params": {"path": str(allowed / ".." / "..")}},
        {"method": "workspace.describe", "params": {"path": "   "}},
        {"method": "workspace.describe", "params": {"path": str(allowed / "sub")}},
    ], roots, db)

    r = results[0]
    check("allowed root accepted", ok(r), str(r)[:150])
    if ok(r):
        data = r["result"]
        check("counts only inside the root", data.get("fileCount") == 2,
              f"fileCount={data.get('fileCount')}")
        check("reports isDirectory", data.get("isDirectory") is True, str(data)[:110])
        check("reports countCapped", data.get("countCapped") is False, str(data)[:110])

    check("outside root refused", err(results[1]), str(results[1])[:150])
    check("traversal refused", err(results[2]), str(results[2])[:150])
    check("empty path refused", err(results[3]), str(results[3])[:150])

    r = results[4]
    check("nested allowed path accepted", ok(r), str(r)[:150])
    if ok(r):
        check("nested count correct", r["result"].get("fileCount") == 1,
              f"fileCount={r['result'].get('fileCount')}")

    # Default containment is home. A path *inside* home is legitimately
    # allowed — tempfile lives under it — so pick a root-sibling to prove the
    # boundary is real rather than trivially permissive.
    system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
    results = call_many(
        [{"method": "workspace.describe", "params": {"path": str(system_root)}}],
        {}, db)
    check("outside home refused by default", err(results[0]), str(results[0])[:170])
    check("refusal names the home boundary",
          "outside the allowed workspace roots" in message(results[0]),
          message(results[0])[:140])

print("=" * 62)
if errors:
    print(f"FAILED: {len(errors)} check(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("ALL WORKSPACE CONTAINMENT CHECKS PASSED")
