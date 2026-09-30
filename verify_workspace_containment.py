"""Live check of workspace.describe containment against DesktopStorage.

desktop_service.py is the reachable caller of describe_workspace, but the
sidecar cannot boot from a repo checkout: it sets ROOT to ``frontend/`` and
then does ``from storage.database import ...``, while the module actually lives
at ``backend/storage/database.py``. That is a pre-existing packaging bug
(untouched by this work, present in the committed original), so this script
drives the same method through a correctly-paired sys.path instead of
importing the sidecar directly.
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


DRIVER = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, {backend!r})   # makes "from storage.database import ..." work
sys.path.insert(0, {repo!r})      # makes "from backend.neuron_inspector import ..." work
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    try:
        from storage.database import DesktopStorage, StorageError
        store = DesktopStorage(Path({db!r}))
        store.initialize()
        data = store.describe_workspace(path=str(req["params"].get("path", "")))
        print(json.dumps({{"ok": True, "data": data}}), flush=True)
    except Exception as exc:
        print(json.dumps({{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}), flush=True)
'''


def call_many(requests, env_extra, db):
    driver = DRIVER.format(backend=str(REPO_ROOT / "backend"),
                           repo=str(REPO_ROOT), db=str(db))
    env = dict(os.environ)
    env.update(env_extra or {})
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        input=payload, capture_output=True, text=True,
        timeout=120, env=env, cwd=str(REPO_ROOT),
    )
    out = []
    for line in proc.stdout.strip().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    if not out:
        return [{"ok": False, "error": proc.stderr[-300:]}]
    return out


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
        {"params": {"path": str(allowed)}},
        {"params": {"path": str(outside)}},
        {"params": {"path": str(allowed / ".." / "..")}},
        {"params": {"path": "   "}},
        {"params": {"path": str(allowed / "sub")}},
    ], roots, db)

    r = results[0]
    check("allowed root accepted", r.get("ok") is True, str(r)[:150])
    if r.get("ok"):
        data = r["data"]
        check("counts only inside the root", data.get("fileCount") == 2,
              f"fileCount={data.get('fileCount')}")
        check("reports isDirectory", data.get("isDirectory") is True, str(data)[:110])
        check("reports countCapped", data.get("countCapped") is False, str(data)[:110])

    check("outside root refused", results[1].get("ok") is False, str(results[1])[:150])
    check("traversal refused", results[2].get("ok") is False, str(results[2])[:150])
    check("empty path refused", results[3].get("ok") is False, str(results[3])[:150])

    r = results[4]
    check("nested allowed path accepted", r.get("ok") is True, str(r)[:150])
    if r.get("ok"):
        check("nested count correct", r["data"].get("fileCount") == 1,
              f"fileCount={r['data'].get('fileCount')}")

    # Default containment is home. A path *inside* home is legitimately
    # allowed — tempfile lives under it — so pick a root-sibling to prove the
    # boundary is real rather than trivially permissive.
    system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
    results = call_many([{"params": {"path": str(system_root)}}], {}, db)
    check("outside home refused by default", results[0].get("ok") is False,
          str(results[0])[:170])
    check("refusal names the home boundary",
          "outside the allowed workspace roots" in str(results[0].get("error", "")),
          str(results[0].get("error"))[:140])

print("=" * 62)
if errors:
    print(f"FAILED: {len(errors)} check(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("ALL WORKSPACE CONTAINMENT CHECKS PASSED")
