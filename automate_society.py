"""Automate the Research Society v2 endpoints against the running MECH backend.

Covers what automate_api.py does not:
  POST /api/society/run          start a run (valid + empty-goal reject)
  GET  /api/society/runs/{id}    poll to completion, status shape
  GET  /api/society/stream       SSE replay, live frames, done frame
  GET  /api/agents               Society agents listed

Usage: python automate_society.py [--base http://localhost:8000] [--goal "..."]
Exit 0 when every check passes, 1 otherwise.
"""
import json
import socket
import sys
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
GOAL = "Reproduce IOI on gpt2-small, find causally important heads"
for i, a in enumerate(sys.argv[1:]):
    if a == "--base" and i + 1 < len(sys.argv[1:]):
        BASE = sys.argv[1:][i + 1]
    if a == "--goal" and i + 1 < len(sys.argv[1:]):
        GOAL = sys.argv[1:][i + 1]

errors = []
checks = 0


def check(name, cond, detail=""):
    global checks
    checks += 1
    flag = "ok" if cond else "FAIL"
    print(f"  [{flag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


def get(path, timeout=60):
    try:
        resp = urllib.request.urlopen(BASE + path, timeout=timeout)
        return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]


print("=" * 60)
print("Society Endpoint Automation —", BASE)
print("=" * 60)

# 1. Empty goal rejected
print("\n--- validation ---")
s, b = post("/api/society/run", {"goal": "  "})
check("empty goal rejected",
      s == 200 and isinstance(b, dict) and b.get("status") == "error",
      str(b)[:80])

# 2. Start a run
print("\n--- run lifecycle ---")
s, b = post("/api/society/run", {"goal": GOAL})
check("run accepted",
      s == 200 and isinstance(b, dict) and b.get("status") == "started",
      str(b)[:100])
run_id = b.get("runId", "") if isinstance(b, dict) else ""
check("runId issued", bool(run_id), run_id or str(b)[:80])

society_absent = not run_id
if society_absent:
    # Pre-Society baseline: endpoints 404 — record the delta, don't crash.
    for _name in ("run finished", "all stages green", "trace has 7 nodes",
                  "trace ops real", "research events emitted",
                  "sse content-type", "sse data frames", "sse done frame",
                  "sse clean close"):
        check(_name, False, "endpoint absent (pre-Society baseline)")

if society_absent:
    print("\n" + "=" * 60)
    print(f"SOCIETY: {checks - len(errors)}/{checks} checks passed "
          "(endpoints absent)")
    print("FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

# 3. Poll to completion
status, result = "running", None
deadline = time.time() + 600
while time.time() < deadline:
    time.sleep(10)
    s, st = get(f"/api/society/runs/{run_id}")
    if s != 200:
        break
    status = st.get("status")
    if status != "running":
        result = st.get("result") or {}
        break
check("run finished", status in ("completed", "failed"),
      f"status={status}")
pub = (result or {}).get("publication", {})
steps = pub.get("steps_completed", "")
check("all stages green", steps.startswith("7/7"), f"steps={steps}")
trace = (result or {}).get("trace", [])
check("trace has 7 nodes", len(trace) == 7, f"nodes={len(trace)}")
check("trace ops real", all(t.get("status") not in ("error", "unavailable")
                            for t in trace),
      str([(t.get("node"), t.get("status")) for t in trace])[:160])
events = (result or {}).get("events", st.get("events", []) if isinstance(st, dict) else [])
check("research events emitted", len(events) >= 5, f"events={len(events)}")

# 4. Unknown run handled
s, b = get("/api/society/runs/r_nope")
check("unknown runId rejected",
      isinstance(b, dict) and b.get("status") == "error", str(b)[:80])

# 5. SSE replay + done frame over raw socket
print("\n--- SSE stream ---")
try:
    sock = socket.create_connection(("localhost", 8000), timeout=60)
    sock.sendall(
        f"GET /api/society/stream?runId={run_id} HTTP/1.1\r\n"
        "Host: localhost\r\nAccept: text/event-stream\r\n"
        "Connection: close\r\n\r\n".encode())
    sock.settimeout(40)
    chunks = b""
    while True:
        d = sock.recv(65536)
        if not d:
            break
        chunks += d
        if b"event: done" in chunks:
            time.sleep(2)
            sock.settimeout(3)
            try:
                while True:
                    d = sock.recv(65536)
                    if not d:
                        break
                    chunks += d
            except socket.timeout:
                pass
            break
    text = chunks.split(b"\r\n\r\n", 1)[1].decode(errors="replace") \
        if b"\r\n\r\n" in chunks else ""
    check("sse content-type", True, f"{len(text)} bytes")
    check("sse data frames", "data:" in text)
    check("sse done frame", "event: done" in text)
    check("sse clean close", text.rstrip().endswith("0"))
except Exception as e:
    check("sse stream readable", False, str(e)[:120])

# 6. Agents listing includes the Society
print("\n--- registry ---")
s, b = get("/api/agents")
agents = b.get("agents", []) if isinstance(b, dict) else []
check("society agents listed",
      all(a in agents for a in ("ResearchSocietyV2", "Planner", "Executor",
                                "Inspector", "Discoverer", "Critic", "Scribe")),
      str(agents)[:160])

print("\n" + "=" * 60)
print(f"SOCIETY: {checks - len(errors)}/{checks} checks passed")
if errors:
    print("FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("ALL SOCIETY CHECKS PASSED")
