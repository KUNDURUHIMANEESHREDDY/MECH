"""Traceable Evidence Graph.

Connects Neuron ➔ Feature ➔ Circuit ➔ Hypothesis ➔ Experiment ➔ Evidence ➔ Publication.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# One vocabulary for provenance, defined once at the boundary.
PROVENANCE_VALUES = {"live", "seeded", "reference", "unavailable"}

EVIDENCE_DIR = os.environ.get("MECH_EVIDENCE_DIR", "backend/storage/evidence")

# Trusted key management is now centralized in provenance.py
from backend.core.provenance import (
    load_trusted_key_ids,
    reset_trusted_key_ids_cache,
)

# Re-export for backwards compatibility
_load_trusted_key_ids = load_trusted_key_ids
_reset_trusted_key_ids_cache = reset_trusted_key_ids_cache


def _is_key_trusted(key_id: Optional[str]) -> bool:
    """Return True iff key_id is in the trusted set."""
    if key_id is None:
        return False
    trusted = load_trusted_key_ids()
    if not trusted:
        return False
    return key_id in trusted


# Numeric step-result keys promoted to Evidence nodes by from_run().
EVIDENCE_KEYS = ("delta", "clean_ld", "patched_ld", "circuit_score",
                 "confidence_score", "attribution_score", "causal_effect",
                 "patch_success_rate", "functional_recovery")


def _iter_evidence(step: Dict[str, Any]):
    """Yield (dotted_key, number) pairs from a trace step, scanning the
    step top level plus one nested dict level (engine results nest)."""
    seen = set()

    def emit(prefix: str, obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            if key in EVIDENCE_KEYS and isinstance(value, (int, float)) \
                    and not isinstance(value, bool):
                dotted = f"{prefix}{key}" if prefix else str(key)
                if dotted not in seen:
                    seen.add(dotted)
                    yield dotted, value

    yield from emit("", step)
    result = step.get("result")
    yield from emit("", result)
    if isinstance(result, dict):
        for key, value in result.items():
            yield from emit(f"{key}.", value)

STEP_TYPES = {"planner": "Plan", "executor": "Execution",
              "inspector": "Observation", "discoverer": "Discovery",
              "critic": "Validation", "scribe": "Publication"}


def _scientific_policy():
    try:
        try:
            from backend.agents.evidence_policy import (
                discovery_is_live,
                provenance_of,
                validation_is_live,
            )
        except ImportError:
            from agents.evidence_policy import (
                discovery_is_live,
                provenance_of,
                validation_is_live,
            )
        return discovery_is_live, provenance_of, validation_is_live
    except Exception:
        return None


def _step_allows_evidence(step: Dict[str, Any]) -> bool:
    """Whether a trace step may contribute Evidence nodes to the graph.

    Inverted from permissive to authorizing: a step yields evidence only when
    it explicitly carries attestation from the measurement layer, declares
    live provenance, sits on a gated node, and its result clears the
    scientific eligibility policy. In particular, executor-shaped steps with
    numeric-looking fields but no attestation — the laundering mechanism —
    never produce Evidence nodes.
    """
    if not isinstance(step, dict):
        return False
    if step.get("attested") is not True:
        return False
    if _step_provenance(step) != "live":
        return False
    node = str(step.get("node", ""))
    if node not in {"discover", "validate"}:
        return False
    policy = _scientific_policy()
    if policy is None:
        return False
    discovery_is_live, _, validation_is_live = policy
    result = step.get("result")
    return (discovery_is_live(result) if node == "discover"
            else validation_is_live(result))


def _step_provenance(step: Dict[str, Any]) -> str:
    """Provenance declared by the step that produced an evidence value.

    Reads the result's own provenance, then the step's. Anything absent or
    unrecognised becomes "unavailable" -- never "live". Defaulting to live here
    is how a seeded measurement became a live-labelled evidence node.
    """
    for holder in (step.get("result"), step):
        if not isinstance(holder, dict):
            continue
        raw = holder.get("provenance")
        if raw is None:
            nested = holder.get("discovery_provenance")
            raw = nested
        if raw is None:
            continue
        label = str(raw).strip().lower()
        if label in PROVENANCE_VALUES:
            return label
        return "unavailable"
    return "unavailable"



class TraceableEvidenceGraph:
    """DAG graph tracking full evidence provenance from neuron activations to final paper."""

    def __init__(self) -> None:
        # Demo-free, matching `from_run` and `empty`.
        #
        # This constructor used to seed a seven-node IOI example -- neuron
        # L8_N402, "SAE #1402", a hypothesis "L8_N402 mediates IOI", an
        # Evidence node labelled "Logit delta -4.2", and "Mechanistic Paper
        # #1" -- plus six edges, for every instance.
        #
        # `AIScientistEngine` builds this class with no arguments and reports
        # `to_dict()["nodes_count"]` in its campaign summary, so a campaign
        # that had measured nothing advertised seven pieces of evidence and a
        # publication. The illustrative graph is still available, but only by
        # asking for it: see `demo`.
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []

    @classmethod
    def demo(cls) -> "TraceableEvidenceGraph":
        """The illustrative IOI graph, for catalog and UI inspection only.

        Every node is marked ``measured`` False with ``provenance``
        "reference", so a consumer cannot rank it as evidence.
        """
        graph = cls()
        graph.nodes = [
            {"id": "n_402", "type": "Neuron", "label": "L8_N402"},
            {"id": "f_1402", "type": "Feature", "label": "SAE #1402"},
            {"id": "c_ioi", "type": "Circuit", "label": "IOI Circuit"},
            {"id": "h_ioi", "type": "Hypothesis", "label": "L8_N402 mediates IOI"},
            {"id": "exp_ioi", "type": "Experiment", "label": "Activation Patching L8_N402"},
            {"id": "ev_ioi", "type": "Evidence", "label": "Logit delta -4.2"},
            {"id": "pub_ioi", "type": "Publication", "label": "Mechanistic Paper #1"},
        ]
        graph.edges = [
            {"source": "n_402", "target": "f_1402", "relation": "encodes"},
            {"source": "f_1402", "target": "c_ioi", "relation": "forms"},
            {"source": "c_ioi", "target": "h_ioi", "relation": "suggests"},
            {"source": "h_ioi", "target": "exp_ioi", "relation": "tested_by"},
            {"source": "exp_ioi", "target": "ev_ioi", "relation": "yields"},
            {"source": "ev_ioi", "target": "pub_ioi", "relation": "substantiates"},
        ]
        for node in graph.nodes:
            node["measured"] = False
            node["provenance"] = "reference"
        return graph

    def get_provenance_trace(self, target_id: str = "pub_ioi") -> List[Dict[str, Any]]:
        return list(self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes_count": len(self.nodes),
            "edges_count": len(self.edges),
            "nodes": self.nodes,
            "edges": self.edges,
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        }

    # -- per-run construction (no demo nodes) --------------------------
    @classmethod
    def empty(cls) -> "TraceableEvidenceGraph":
        graph = cls.__new__(cls)
        graph.nodes = []
        graph.edges = []
        return graph

    def add_node(self, id: str, type: str, label: str,
                 payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        for existing in self.nodes:
            if existing.get("id") == id:
                return existing
        node: Dict[str, Any] = {"id": id, "type": type, "label": label}
        if payload:
            node["payload"] = payload
        self.nodes.append(node)
        return node

    def add_edge(self, source: str, target: str,
                 relation: str) -> Dict[str, Any]:
        edge = {"source": source, "target": target, "relation": relation}
        if edge not in self.edges:
            self.edges.append(edge)
        return edge

    @classmethod
    def from_run(cls, run_id: str, goal: str,
                 trace: List[Dict[str, Any]]) -> "TraceableEvidenceGraph":
        """Build a graph from a real Society run trace (demo-free)."""
        graph = cls.empty()
        goal_id = f"goal_{run_id}"
        graph.add_node(goal_id, "Goal", goal[:80])
        prev = goal_id
        for step in trace:
            node_id = f"{run_id}_{step.get('node', 'step')}"
            agent = str(step.get("agent", ""))
            status = str(step.get("status", ""))
            node_payload = {
                "agent": agent,
                "status": status,
                "op": step.get("op", ""),
            }
            if str(step.get("node", "")) in {"discover", "validate"}:
                policy = _scientific_policy()
                if policy is not None:
                    _, provenance_of, _ = policy
                    node_payload["provenance"] = provenance_of(step.get("result"))
                    node_payload["scientific_eligible"] = _step_allows_evidence(step)
            graph.add_node(
                node_id, STEP_TYPES.get(agent, "Execution"),
                f"{step.get('node', 'step')} ({status})", node_payload)
            graph.add_edge(prev, node_id, "followed_by")
            if _step_allows_evidence(step):
                for key, value in _iter_evidence(step):
                    ev_id = f"{node_id}_ev_{key.replace('.', '_')}"
                    # Provenance is derived, never defaulted to live. This used
                    # to initialise `evidence_provenance = "live"` and only
                    # overwrite it when a result happened to carry a
                    # provenance key -- so a step whose result omitted the key
                    # silently produced a live-labelled evidence node. Steps
                    # other than discover/validate skip the scientific
                    # eligibility gate entirely, which is how a seeded
                    # measurement could reach the graph labelled live.
                    evidence_provenance = _step_provenance(step)
                    node_payload = {
                        "value": value,
                        "provenance": evidence_provenance,
                        "field_provenance": {"value": evidence_provenance},
                    }
                    # An unattested "live" is not evidence. Only a step that
                    # both declares live and passes its eligibility gate may
                    # produce a live-labelled node.
                    if evidence_provenance == "live":
                        node_payload["attested"] = False
                        node_payload["reason"] = (
                            "Declared live by the producing step but not "
                            "attested with a RunAttestation; treated as "
                            "unverified."
                        )
                    graph.add_node(
                        ev_id,
                        "Evidence",
                        f"{key} = {value}",
                        node_payload,
                    )
                    graph.add_edge(node_id, ev_id, "yields")
            prev = node_id
        return graph

    # -- JSON file persistence (ad-hoc graph exports, NOT the authoritative
    # run path — Society records go through save_run_record below) --------
    def save(self, path: str) -> str:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "TraceableEvidenceGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = cls.empty()
        graph.nodes = data.get("nodes", [])
        graph.edges = data.get("edges", [])
        return graph


def _evidence_dir(directory: Optional[str] = None) -> str:
    return directory or EVIDENCE_DIR


#: Run IDs are minted as `"r" + uuid4().hex[:12]` in
#: `backend.api.dispatcher.society_run`. Anything else — `..`, absolute
#: paths, UNC shares, drive-qualified names, URL-encoded traversal — is not
#: a run ID and is rejected before the filesystem is touched.
RUN_ID_RE = re.compile(r"^r[0-9a-f]{12}$")


def validate_run_id(run_id: Any) -> str:
    """Return `run_id` iff it is a structurally valid Society run ID.

    Raises :class:`ValueError` for everything else, so a path-like `run_id`
    can never become a filename.
    """
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")
    return run_id


def _record_path(run_id: str, directory: Optional[str] = None) -> str:
    """Resolve the evidence file for a validated run ID, contained in-root.

    Defense in depth: the regex above already makes traversal structurally
    impossible, and this still verifies the resolved path stays inside the
    evidence root before any open().
    """
    validate_run_id(run_id)
    root = Path(_evidence_dir(directory)).resolve()
    target = (root / f"{run_id}.json").resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"run_id escapes the evidence directory: {run_id!r}")
    return str(target)


#: Serializes evidence commits across processes and threads. Same-target
#: os.replace calls race on Windows file locking (WinError 5) and on POSIX
#: without advisory locking. Uses a cross-process file lock that works on
#: both Windows (LockFileEx) and POSIX (fcntl.flock).
import threading
import os
import sys

_save_lock = threading.Lock()  # process-local fast path
_cross_process_lock_path = None  # set on first use

def _acquire_cross_process_lock(target_dir: str) -> tuple:
    """Acquire a cross-process exclusive lock on the evidence directory.
    Returns (lock_handle, lock_path) or raises on failure.
    Uses non-blocking locks with retry to avoid deadlocks in multi-threaded scenarios.
    """
    import os
    import time
    lock_path = os.path.join(target_dir, ".evidence_write.lock")
    deadline = time.monotonic() + 30.0  # 30 second timeout
    
    if sys.platform == "win32":
        import msvcrt
        # On Windows, use non-blocking lock with retry
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_BINARY)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return fd, lock_path
            except OSError as e:
                # On Windows, PermissionError (13) means lock held by another process
                # EDEADLK (36) means deadlock - retry
                if e.errno not in (13, 36) or time.monotonic() >= deadline:
                    os.close(fd)
                    raise
                time.sleep(0.01)  # 10ms backoff
    else:
        import fcntl
        # On POSIX, use non-blocking flock with retry
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd, lock_path
            except OSError as e:
                if e.errno not in (11, 35) or time.monotonic() >= deadline:  # EAGAIN, EWOULDBLOCK
                    os.close(fd)
                    raise
                time.sleep(0.01)  # 10ms backoff


def _release_cross_process_lock(fd: int, lock_path: str) -> None:
    """Release a cross-process lock acquired with _acquire_cross_process_lock."""
    import os
    import sys
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
        try:
            os.unlink(lock_path)
        except OSError:
            pass


#: Serializes evidence commits in this process. Same-target os.replace calls
#: race on Windows file locking (WinError 5); production saves once per run
#: from one worker thread, but the lock makes last-wins deterministic
#: everywhere. Cross-process races are prevented by a file-based lock.
_save_lock = threading.Lock()


#: Version of the signed envelope written by `save_run_record`. There are no
#: prior envelope versions; v1 is the first.
ENVELOPE_SCHEMA_VERSION = 1

#: Domain separator binding a signature to this artifact type and version.
_ENVELOPE_PAYLOAD_PREFIX = "mech-run-record-v1"


def _canonical_record_json(record: Dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      default=str)


def _signing_payload(run_id: str, record: Dict[str, Any], *,
                    created_at: str, public_key: str) -> str:
    return ("\n".join((_ENVELOPE_PAYLOAD_PREFIX,
                       str(ENVELOPE_SCHEMA_VERSION), run_id, created_at,
                       public_key, _canonical_record_json(record))))


def _attest_envelope(run_id: str, record: Dict[str, Any],
                     created_at: str) -> Dict[str, Any]:
    """Build the attestation block for a run record envelope.

    Uses the real Ed25519 module (`backend.science.integrity.signing`), which
    never defaults a key. With no key configured the envelope is written
    unsigned (`attested: False`) with the reason stated — readable, but never
    mistaken for tamper-evident. A signature here proves only that *this
    backend wrote these bytes*; it says nothing about whether the numbers
    inside are real measurements. That is the eligibility layer's job.
    """
    try:
        from backend.science.integrity import signing as _signing
        public_key = _signing.public_key_hex()
        signature = _signing.sign(
            _signing_payload(run_id, record, created_at=created_at,
                             public_key=public_key))
        return {
            "attested": True,
            "algorithm": "Ed25519",
            "public_key": public_key,
            "key_id": _signing.key_id_for(public_key),
            "signature": signature,
            "reason": None,
        }
    except Exception as exc:
        reason = ("no signature was recorded: "
                  f"{type(exc).__name__}: {exc}")[:300]
        return {
            "attested": False,
            "algorithm": "Ed25519",
            "public_key": None,
            "key_id": None,
            "signature": None,
            "reason": reason,
        }


def _is_envelope(data: Any) -> bool:
    return (isinstance(data, dict)
            and data.get("schema_version") == ENVELOPE_SCHEMA_VERSION
            and isinstance(data.get("record"), dict))


def _is_historical(data: Any) -> bool:
    """True when a record carries the quarantine stamp (copied back or not)."""
    return isinstance(data, dict) and data.get("origin") == HISTORICAL_ORIGIN


def envelope_status(run_id: str,
                    directory: Optional[str] = None) -> Dict[str, Any]:
    """Verify the persistence envelope for a run without serving its record.

    Returns a status dict; never raises. `"legacy"` means the file predates
    envelopes and is served unverified (quarantine decides its fate, not this
    function). `"valid"` means the bytes verify under the embedded key AND the
    signing key is in the configured trusted set (MECH_TRUSTED_SIGNING_KEY_PATH)
    AND, when the embedded record claims measurement attestation
    (`attested is True` or a `measurement_record` is present), that claim
    itself verifies (trusted-key signature, weight-derived identity,
    recomputed commitments). Envelopes with valid signatures but untrusted keys get
    `attested: False, signature: "untrusted"`. A forged measurement claim
    verifies as unattested even though the envelope bytes are authentic:
    the envelope proves this backend wrote the bytes, not that the bytes
    are evidence. Callers that pin a key must
    additionally compare `key_id`.
    """
    try:
        path = _record_path(run_id, directory)
    except ValueError as exc:
        return {"envelope": "invalid", "attested": False,
                "signature": "missing", "reason": str(exc)[:200]}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"envelope": "missing", "attested": False,
                "signature": "missing", "reason": "no record on disk"}
    except Exception as exc:
        return {"envelope": "corrupt", "attested": False,
                "signature": "missing",
                "reason": f"unreadable record: {type(exc).__name__}"[:200]}
    if not _is_envelope(data):
        return {"envelope": "legacy", "attested": False,
                "signature": "missing",
                "reason": "record predates signed envelopes; unverified"}
    if _is_historical(data) or _is_historical(data.get("record")):
        return {"envelope": "v1", "attested": False,
                "signature": "invalid",
                "reason": "record carries the quarantine stamp"}
    if data.get("run_id") != run_id:
        return {"envelope": "v1", "attested": False,
                "signature": "invalid",
                "reason": "envelope run_id does not match the filename"}
    attestation = data.get("attestation")
    if not isinstance(attestation, dict) or not attestation.get("attested"):
        return {"envelope": "v1", "attested": False,
                "signature": "missing",
                "reason": str((attestation or {}).get("reason")
                              or "record was saved without a signature")}
    signature = attestation.get("signature")
    public_key = attestation.get("public_key")
    if not signature or not public_key:
        return {"envelope": "v1", "attested": False,
                "signature": "missing",
                "reason": "attestation claims attested but carries no signature/key"}
    if attestation.get("algorithm") != "Ed25519":
        return {"envelope": "v1", "attested": False,
                "signature": "invalid",
                "reason": "unsupported attestation algorithm"}
    try:
        from backend.science.integrity import signing as _signing
        result = _signing.verify(
            _signing_payload(run_id, data["record"],
                             created_at=str(data.get("created_at", "")),
                             public_key=str(public_key)),
            signature, public_key)
    except Exception as exc:
        return {"envelope": "v1", "attested": False,
                "signature": "unverifiable",
                "reason": f"verifier failed: {type(exc).__name__}"[:200]}
    if not result.valid:
        return {"envelope": "v1", "attested": False,
                "signature": "invalid",
                "reason": str(result.reason or "signature does not verify")[:200]}
    # Signature is valid; now check if the signing key is TRUSTED.
    key_id = attestation.get("key_id")
    if not _is_key_trusted(key_id):
        return {"envelope": "v1", "attested": False,
                "signature": "untrusted",
                "reason": ("signature verifies but the signing key is not in the "
                           "configured trusted key set")[:200],
                "key_id": key_id}
    # The envelope proves this backend wrote these bytes. That alone says
    # nothing about whether the bytes are measurement evidence, so a
    # record that CLAIMS measurement attestation must clear it: trusted-key
    # signature on its measurement record, weight-derived model identity,
    # and commitments that recompute from the embedded record. A forged
    # `attested: True` boolean with no (or a bogus) measurement record
    # saves fine but verifies as unattested here. Plain records that claim
    # nothing keep the envelope-level verdict.
    embedded = data["record"]
    claims_measurement = (isinstance(embedded, dict)
                          and (embedded.get("attested") is True
                               or "measurement_record" in embedded))
    if claims_measurement:
        try:
            from backend.agents import evidence_policy as _policy
            embedded_ok = _policy._is_attested(embedded)
        except Exception:
            embedded_ok = False
        if not embedded_ok:
            return {"envelope": "v1", "attested": False,
                    "signature": "valid",
                    "reason": ("envelope signature verifies under a trusted "
                               "key but the embedded record is not valid "
                               "measurement evidence")[:200],
                    "key_id": key_id}
    return {"envelope": "v1", "attested": True, "signature": "valid",
            "key_id": key_id, "reason": None}


#: How long `_replace_with_retry` keeps retrying a sharing-violation rename.
_REPLACE_RETRY_SECONDS = 5.0

#: Pause between attempts. Windows releases the target within milliseconds;
#: the bound exists so a genuinely stuck writer fails the commit instead of
#: hanging the suite forever.
_REPLACE_RETRY_BACKOFF = 0.005


def _replace_with_retry(src: str, dst: str) -> None:
    """`os.replace`, retried while Windows reports the target as busy.

    The WinError 5 this absorbs is a *sharing violation*, not a permissions
    problem. `os.replace` deletes the destination and renames in one step; on
    Windows a concurrent reader or writer holding the destination open makes
    that delete fail with `PermissionError` (WinError 5). The window is short
    -- another process's open handle, which closes as soon as it is done --
    so retrying turns a spurious failure into the documented last-wins result.

    Why this is needed here despite `_save_lock`: that lock is a
    `threading.Lock`, so it serialises commits only within one process. Two
    agent processes -- or the two-process durability tests -- never touch it.
    Measured before this was added:
    `test_concurrent_processes_same_run_id_no_corruption` failed 3 of 6 runs
    with `Access is denied` on the rename.

    The lock is kept. It is still correct and still cheap for the common
    single-process case; it simply does not cover this.
    """
    import time

    deadline = time.monotonic() + _REPLACE_RETRY_SECONDS
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_REPLACE_RETRY_BACKOFF)


def save_run_record(run_id: str, record: Dict[str, Any],
                    directory: Optional[str] = None) -> str:
    """Persist a full Society run record (trace + graph + publication).

    The record is wrapped in a versioned, Ed25519-signed envelope (or an
    explicitly unsigned one when no signing key is configured — see
    `_attest_envelope`). Use `envelope_status` to check a file; `load`
    unwraps but never upgrades.
    """
    from datetime import datetime, timezone

    validate_run_id(run_id)
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    if record.get("origin") == HISTORICAL_ORIGIN:
        raise ValueError("refusing to re-mint a quarantined record as current evidence")
    target = _evidence_dir(directory)
    os.makedirs(target, exist_ok=True)
    path = _record_path(run_id, directory)
    created_at = datetime.now(timezone.utc).isoformat()
    envelope = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "record": record,
        "attestation": _attest_envelope(run_id, record, created_at),
    }
    # Atomic commit: readers never see a partially written record. The payload
    # goes to a uniquely named sidecar temp in the same directory (same
    # filesystem, so the rename swaps the directory entry atomically — a
    # concurrent reader sees the old file or the new one, never torn bytes),
    # is flushed and fsynced, then moved over the target with os.replace.
    # Crash/power-loss durability of the rename itself is best-effort
    # (_fsync_dir is a no-op on Windows, where opening a directory fails).
    # The commit holds _save_lock because same-target replaces race on
    # Windows file locking (measured WinError 5 under a 10-thread hammer);
    # last-wins stays deterministic. A crash can only orphan a sidecar, which
    # readers ignore (list only reads *.json).
    import tempfile
    # Cross-process lock (outer) + thread lock (inner) for full serialization
    cross_fd, cross_lock_path = _acquire_cross_process_lock(target)
    try:
        with _save_lock:
            fd, tmp_path = tempfile.mkstemp(dir=target, prefix=f"{run_id}.json.tmp.")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(envelope, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                # Rename first, then fsync directory to persist the directory entry.
                # This order ensures the file data is durable (fsynced before rename),
                # and the directory entry update is persisted after the rename.
                _replace_with_retry(tmp_path, path)
                try:
                    _fsync_dir(target)
                except OSError:
                    pass
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
    finally:
        _release_cross_process_lock(cross_fd, cross_lock_path)
    return path


def _fsync_dir(directory: str) -> None:
    """Persist the directory entry itself (rename durability). Best effort."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_run_record(run_id: str,
                    directory: Optional[str] = None) -> Dict[str, Any]:
    """Load a run record, unwrapping the persistence envelope if present.

    Pre-envelope (legacy) files are returned as-is; check `envelope_status`
    to tell the two apart. Unwrapping never upgrades: a legacy record stays
    legacy until something re-saves it through `save_run_record`. Anything
    carrying the quarantine stamp is refused: a stamped file copied back
    into the authoritative dir must not pass as current evidence.
    """
    path = _record_path(run_id, directory)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if _is_historical(data):
        raise ValueError(f"quarantined record is not servable: {run_id!r}")
    if _is_envelope(data):
        record = data["record"]
        if _is_historical(record):
            raise ValueError(f"quarantined record is not servable: {run_id!r}")
        return record
    return data


#: Most runs returned by `list_run_records`. Applied AFTER ordering by
#: `created_at`, so a directory holding more than this keeps its newest runs
#: rather than an arbitrary subset.
LIST_LIMIT = 200


def _creation_sort_key(summary: Dict[str, Any]):
    """Sort key for newest-first listing: dated records first, undated last.

    A run with no usable timestamp is not the newest run, so it sorts behind
    every dated one instead of being guessed at. `run_id` is the tiebreak, so
    the order is stable for records written in the same instant.
    """
    run_id = str(summary.get("run_id", ""))
    raw = summary.get("created") or ""
    if not raw:
        return (1, 0.0, run_id)
    try:
        moment = _dt.datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return (1, 0.0, run_id)
    if moment.tzinfo is None:
        # A naive stamp is compared as UTC rather than as local time, so the
        # ordering does not shift with the machine's timezone.
        moment = moment.replace(tzinfo=_dt.timezone.utc)
    return (0, -moment.timestamp(), run_id)


def list_run_records(directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """Summaries of persisted runs, newest first. Never raises.

    Ordering is by each run's `created_at`, not by filename. A run's filename
    is its run_id, and run ids are minted as ``"r" + uuid4().hex[:12]`` --
    twelve random hex characters carrying no time at all. The previous
    ``sorted(filenames, reverse=True)`` therefore reverse-alphabetised random
    strings and produced an arbitrary order beneath a "newest first" docstring.

    The envelope's top-level `created_at` is the value used, because the
    attestation signature binds it (`_signing_payload` takes `created_at` as an
    input), so it cannot be edited without invalidating verification.
    """
    target = _evidence_dir(directory)
    try:
        names = [f for f in os.listdir(target) if f.endswith(".json")]
    except Exception:
        return []

    summaries = []
    for name in names:
        try:
            with open(os.path.join(target, name), "r",
                      encoding="utf-8") as f:
                record = json.load(f)
            if _is_envelope(record):
                outer_id = record.get("run_id", name[:-5])
                created_at = str(record.get("created_at", "") or "")
                record = record["record"]
            else:
                # A pre-envelope file has no top-level creation time. It still
                # lists, and sorts behind every dated run.
                outer_id = None
                created_at = ""
            if not isinstance(record, dict):
                continue
            if _is_historical(record):
                continue  # quarantine stamp: never listed as a current run
            pub = record.get("publication", {})
            summaries.append({
                "run_id": outer_id or record.get("run_id", name[:-5]),
                "goal": record.get("goal", ""),
                "status": record.get("status", ""),
                "steps_completed": pub.get("steps_completed", ""),
                # `created_at` is the moment the ordering is by. It is not
                # `published_at`: that field is absent for any run which has
                # not been published, so reading it here left `created` blank
                # for exactly the runs a caller most wants to see.
                "created": created_at or str(record.get("created_at", "") or ""),
                "published": str(pub.get("published_at", "") or ""),
            })
        except Exception:
            continue

    summaries.sort(key=_creation_sort_key)
    return summaries[:LIST_LIMIT]


#: Origin marker stamped on every quarantined record. Directory membership in
#: the historical dir is the quarantine mark; the stamp is load-bearing too:
#: `load_run_record` and `list_run_records` refuse anything carrying it, so
#: a file copied back cannot pass as current evidence.
HISTORICAL_ORIGIN = "historical_fixture"

#: Eligibility flags forced False on quarantine, top level and inside the
#: result dict. Consumers gating on evidence_policy see a closed gate.
_QUARANTINE_FLAGS = ("scientific_eligible", "validation_eligible",
                     "publication_eligible")


def _historical_dir(directory: Optional[str] = None) -> str:
    override = os.environ.get("MECH_EVIDENCE_HISTORICAL_DIR", "").strip()
    if override:
        return override
    return os.path.join(os.path.dirname(_evidence_dir(directory) or ".") or ".",
                        "evidence_historical")


def _stamp_historical(record: Any) -> Any:
    """Mark a record as a historical fixture, closing every eligibility gate."""
    if not isinstance(record, dict):
        return record
    record["origin"] = HISTORICAL_ORIGIN
    for flag in _QUARANTINE_FLAGS:
        record[flag] = False
    result = record.get("result")
    if isinstance(result, dict):
        for flag in _QUARANTINE_FLAGS:
            result[flag] = False
    return record


def quarantine_legacy_records(directory: Optional[str] = None,
                              historical_directory: Optional[str] = None) -> Dict[str, Any]:
    """Move pre-envelope records out of the authoritative evidence directory.

    Idempotent one-time migration (also run at backend startup): every
    ``*.json`` file that is not a v1 envelope is stamped
    ``origin: historical_fixture`` with all eligibility flags forced False
    and moved to the historical directory. Envelope records, ``*.tmp.*``
    sidecars, and anything else are untouched. Never raises; reports counts.
    """
    target = _evidence_dir(directory)
    dest = historical_directory or _historical_dir(directory)
    report: Dict[str, Any] = {"moved": 0, "stamped": 0, "skipped": 0,
                               "destination": dest, "failures": []}
    try:
        if Path(target).resolve() == Path(dest).resolve():
            report["reason"] = "historical directory is the evidence directory; refusing"
            return report
    except OSError:
        pass
    try:
        names = os.listdir(target)
    except Exception as e:
        report["failures"].append({"stage": "listdir", "error": str(e)})
        return report
    for name in names:
        if not name.endswith(".json"):
            continue
        src = os.path.join(target, name)
        try:
            with open(src, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            report["failures"].append({"filename": name, "stage": "read", "error": str(e)})
            continue
        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            report["failures"].append({"filename": name, "stage": "json_decode", "error": str(e)})
            data = None
        if _is_envelope(data):
            report["skipped"] += 1
            continue
        try:
            os.makedirs(dest, exist_ok=True)
            if isinstance(data, dict):
                _stamp_historical(data)
                report["stamped"] += 1
                payload: bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
            else:
                # Unparseable bytes move verbatim: forensics over formatting.
                payload = raw_bytes
            tmp = os.path.join(dest, name + f".tmp.{os.getpid()}")
            with open(tmp, "wb") as f:
                f.write(payload)
            os.replace(tmp, os.path.join(dest, name))
            os.unlink(src)
            report["moved"] += 1
        except Exception as e:
            report["failures"].append({"filename": name, "stage": "write_move", "error": str(e)})
            continue
    return report


def _quarantined_path(run_id: str, directory: Optional[str] = None,
                      historical_directory: Optional[str] = None) -> str:
    validate_run_id(run_id)
    root = Path(historical_directory or _historical_dir(directory)).resolve()
    target = (root / f"{run_id}.json").resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"run_id escapes the quarantine directory: {run_id!r}")
    return str(target)


def load_quarantined_record(run_id: str,
                            directory: Optional[str] = None,
                            historical_directory: Optional[str] = None
                            ) -> Dict[str, Any]:
    """Read a quarantined record for audit. Never served as a live run."""
    path = _quarantined_path(run_id, directory, historical_directory)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_QUARANTINED_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.json$")


def load_quarantined_file(filename: str,
                          directory: Optional[str] = None,
                          historical_directory: Optional[str] = None) -> Any:
    """Read a quarantined file by exact filename (covers odd legacy names).

    The name must be a bare ``*.json`` filename — no separators, no
    traversal — and the file must resolve inside the quarantine directory.
    """
    if (not isinstance(filename, str) or not _QUARANTINED_FILENAME_RE.match(filename)
            or "/" in filename or "\\" in filename):
        raise ValueError(f"invalid quarantined filename: {filename!r}")
    root = Path(historical_directory or _historical_dir(directory)).resolve()
    target = (root / filename).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"filename escapes the quarantine directory: {filename!r}")
    with open(str(target), "r", encoding="utf-8") as f:
        return json.load(f)


def list_quarantined_records(directory: Optional[str] = None,
                             historical_directory: Optional[str] = None
                             ) -> List[Dict[str, Any]]:
    """Summaries of quarantined records, flagged by origin. Never raises."""
    target = historical_directory or _historical_dir(directory)
    try:
        files = sorted(
            (f for f in os.listdir(target) if f.endswith(".json")),
            reverse=True)
    except Exception:
        return []
    summaries = []
    for name in files[:200]:
        try:
            with open(os.path.join(target, name), "r",
                      encoding="utf-8") as f:
                record = json.load(f)
        except Exception:
            summaries.append({
                "run_id": name[:-5] if name.endswith(".json") else name,
                "goal": "",
                "status": "",
                "origin": "unparseable",
            })
            continue
        try:
            if not isinstance(record, dict):
                continue
            summaries.append({
                "run_id": record.get("run_id", name[:-5]),
                "goal": record.get("goal", ""),
                "status": record.get("status", ""),
                "origin": record.get("origin", HISTORICAL_ORIGIN),
            })
        except Exception:
            continue
    return summaries
