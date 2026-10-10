"""Capture real, measured interpretability results from live GPT-2 weights.

Every number written by this script comes from a forward pass over real GPT-2
small weights via ``transformers``. Nothing is simulated, sampled from a
distribution, or copied from a literature baseline. The output is what the
README quotes as results.

Nothing in the artifact is a literal either
-------------------------------------------
This script used to carry `"model": "gpt2 (124M)"` in its ``meta`` block and to
hardcode the layers it inspected (``list_neurons(8, ...)``, ``head_detail(9, 9)``,
steering layers ``(6, 8, 10)``). Three problems, in increasing order of how long
they hide:

* The artifact described weights that were never checked. If the engine loaded
  a different checkpoint, the report still announced GPT-2 small.
* The inspected layers were baked into both the code and the *result keys*
  (``top_neurons_L8_by_in_norm``, ``head_L9H9``), so a checkpoint with a
  different depth would have measured the wrong layers under a label claiming
  otherwise.
* The reader could not tell a measurement from a constant.

So identity is now a capture. `capture_identity` reads the loaded model and
tokenizer, hashes them, records the checkpoint commit, and is a *required*
section: without it the run cannot reach `provenance: "live"`. Layer and head
choices are stated as fractions of the loaded depth and resolved against the
architecture, then recorded in the artifact. The only numbers still written by
hand are the experiment *inputs* -- the IOI templates and the steering prompt
pair -- which are choices rather than findings, and they are hashed and recorded
so they can be told apart.

Fail-closed by construction
---------------------------
A capture that raises must not be able to leave this script with no artifact at
all. Previously `main()` ran each capture as a bare statement:

    data["ioi"] = timed("ioi (3 templates)", capture_ioi)

so an exception in `capture_ioi()` propagated out of `main()`, and the
provenance derivation designed to report `unavailable` on an incomplete run
never executed -- there was no `capture.json` to carry the label. The comment
promised fail-closed behaviour the control flow could not deliver.

Every capture now runs inside `run_section`, which catches, records, and
continues. A failed section is *represented* in the artifact (status, timings,
error type and message) rather than erasing the run, the measurement key is
absent from `data` so the presence-based derivation withholds `live`, and the
process exits non-zero.

`json.dump(..., default=str)` is also gone. It had not corrupted anything the
committed artifact contains -- the engine returns Python floats, so the
fallback never fired -- but it reported nothing when it did convert, which is
the property worth removing. Payloads are made JSON-native explicitly, and an
unrecognised type raises instead of being stringified into something that parses
as JSON and reads as a measurement. See `to_json_native`.

Outputs
-------
docs/results/capture.json    structured measurements, plus a per-section run report
docs/images/*.png            figures rendered from validated measurements only

Run:
    python scripts/capture_results.py
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

from backend.services import gpt2_engine as engine  # noqa: E402

OUT_DIR = os.path.join(ROOT, "docs", "results")
IMG_DIR = os.path.join(ROOT, "docs", "images")

# Standard IOI (Indirect Object Identification) template from
# Wang et al. 2022, "Interpretability in the Wild: a Circuit for
# Indirect Object Identification in GPT-2 Small" (arXiv:2211.00593).
#
# These are experiment *inputs*, not findings, and are the one class of value
# that legitimately stays written by hand: a published template is a choice, and
# changing it would change the science. They are hashed and recorded in the
# artifact so a reader can tell a declared input from a measured result.
IOI_TEMPLATES = [
    ("Then John and Mary went to the store. John gave a bottle of milk to", " Mary", " John"),
    ("Then John and Mary went to the park. John gave a bottle of milk to", " Mary", " John"),
    ("Then John and Mary went to the office. John gave a key to", " Mary", " John"),
]

CAPITAL_PROMPT = "The capital of France is"
STEER_POS = "The capital of France is"
STEER_NEG = "The capital of Japan is"

#: Steering coefficients swept at every chosen layer. Declared, like the prompts.
STEER_ALPHAS = (5.0, 10.0, 20.0, 40.0, 80.0)

#: Sanity checks: name, prompt, and the token GPT-2 must put first for the check
#: to mean anything. The expectation is the claim being tested, and it is
#: reported with a `passes` flag rather than asserted -- so a checkpoint that
#: behaves differently is recorded, not hidden.
SANITY_CHECKS = (
    ("induction", "Hello, my name is Julien. Hello, my name is", "Jul",
     "repeats the name stem ' Jul'"),
    ("counting", "1, 2, 3, 4, 5, 6,", "7", "continues with ' 7'"),
)

# --------------------------------------------------------------------------
# Selection rules
#
# Which layers to inspect is a choice, but the *index* is not a fact about the
# world -- it is a fact about a particular checkpoint. These were hardcoded as
# `8`, `9`/`9` and `(6, 8, 10)`, which meant a different checkpoint would have
# been measured at the same indices and reported under keys literally named
# after them.
#
# So each is stated as a fraction of the loaded depth or width and resolved
# against the architecture. At GPT-2 small (12 layers, 12 heads) they resolve to
# exactly the indices that were hardcoded -- steering (6, 8, 10), inspection
# layer 8, head (9, 9) -- so the published measurements are unchanged, and the
# resolved indices are recorded in the artifact.
# --------------------------------------------------------------------------

STEERING_LAYER_FRACTIONS = (0.5, 2 / 3, 5 / 6)

#: The neuron layer and the head layer are *different* layers. This distinction
#: was nearly lost: an earlier version of this file resolved one pair and used it
#: for both, quietly moving the head inspection from layer 9 to layer 8 while
#: every other number stayed identical. Kept as separate rules so it cannot
#: collapse again.
INSPECTION_NEURON_LAYER_FRACTION = 2 / 3   # was list_neurons(8, ...)
INSPECTION_HEAD_LAYER_FRACTION = 3 / 4     # was head_detail(9, 9)
INSPECTION_HEAD_FRACTION = 3 / 4           # was head_detail(9, 9)


def _resolve_fractions(fractions, extent: int, what: str) -> Tuple[int, ...]:
    """Turn depth fractions into concrete indices, refusing to clamp.

    Clamping would silently turn a mis-specified rule into a valid-looking
    index -- and at `extent == 1` every fraction collapses onto index 0, so a
    three-layer "sweep" would report the same layer three times and look like a
    result. Raising is the honest response: the rule is wrong for this model.
    """
    resolved = [int(round(fraction * extent)) for fraction in fractions]
    for fraction, index in zip(fractions, resolved):
        if not 0 <= index < extent:
            raise ValueError(
                f"{what}: fraction {fraction} resolves to index {index}, "
                f"outside a depth of {extent}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            f"{what}: fractions {fractions} collapse onto {sorted(set(resolved))} "
            f"at a depth of {extent}; they would measure the same index twice")
    return tuple(resolved)


def steering_layers(n_layers: int) -> Tuple[int, ...]:
    return _resolve_fractions(STEERING_LAYER_FRACTIONS, n_layers, "steering layers")


def inspection_indices(n_layers: int, n_heads: int) -> Tuple[int, int, int]:
    """`(neuron_layer, head_layer, head)` for the loaded model.

    Two layers, not one. The neuron and head inspections were `list_neurons(8,
    ...)` and `head_detail(9, 9)` -- different layers, and reusing one index for
    both would move a measurement while changing nothing else an artifact shows.
    """
    neuron_layer = _resolve_fractions(
        (INSPECTION_NEURON_LAYER_FRACTION,), n_layers, "inspection neuron layer")[0]
    head_layer = _resolve_fractions(
        (INSPECTION_HEAD_LAYER_FRACTION,), n_layers, "inspection head layer")[0]
    head = _resolve_fractions(
        (INSPECTION_HEAD_FRACTION,), n_heads, "inspection head")[0]
    return neuron_layer, head_layer, head


#: The architecture this experiment protocol was written for.
#:
#: This is a *check*, not a source of reported values -- every number in the
#: artifact comes from the weights. It exists because the protocol itself is
#: GPT-2-specific: the IOI circuit, and the induction and counting behaviours it
#: sanity-checks. A run against different weights would produce
#: differently-parameterised numbers still labelled as this protocol's results.
#: Rejecting is the honest response; see `assert_protocol_applicable`.
PROTOCOL_REFERENCE_ARCHITECTURE = {
    "model_id": "gpt2",
    "n_layers": 12,
    "n_heads": 12,
    "d_model": 768,
    "vocab_size": 50257,
}


# --------------------------------------------------------------------------
# Fail-closed capture runner
# --------------------------------------------------------------------------

def to_json_native(value: Any, path: str = "$") -> Any:
    """Convert `value` into something `json.dump` accepts without `default=`.

    `json.dump(..., default=str)` used to sit on the artifact write. To be
    precise about what it did and did not do: the artifact contains no
    stringified numbers -- every `delta` and `effect` in it is a JSON number,
    because the engine happens to return Python floats today. So this is not a
    repair of a visible corruption.

    It is the removal of a *silent* fallback. `default=` converts anything the
    encoder cannot handle by calling `str()` on it and reporting nothing. The
    day a capture starts returning a tensor instead of a float -- a plausible
    refactor, one `.item()` away -- the artifact would carry
    `"tensor(1.5011, grad_fn=<AddBackward0>)"` where a measurement belongs,
    still labelled `provenance: "live"` and `measured_sections: "8/8"`. The run
    would report success and the artifact would read as evidence. Every
    conversion here is explicit, and an unrecognised type raises `TypeError`
    naming the path where it was found, so the failure is loud and located.

    Non-finite floats are rejected too, and that one is demonstrable rather than
    hypothetical: `json.dump` writes bare `NaN` by default, which is not valid
    JSON, so a single NaN measurement produces a file Python reads and another
    language's parser rejects.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"non-finite float at {path}: {value!r}")
        return value
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            name = key if isinstance(key, str) else str(key)
            out[name] = to_json_native(item, f"{path}.{name}")
        return out
    if isinstance(value, (list, tuple)):
        return [to_json_native(item, f"{path}[{i}]") for i, item in enumerate(value)]

    # numpy scalar: 0-d, carries .item()
    shape = getattr(value, "shape", None)
    item = getattr(value, "item", None)
    if shape == () and callable(item):
        return to_json_native(item(), path)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_json_native(tolist(), path)
    if callable(item):
        return to_json_native(item(), path)

    raise TypeError(
        f"unserialisable {type(value).__name__} at {path}; refusing to write "
        f"its repr into the artifact"
    )


#: Keys each section must carry for its payload to count as a measurement.
#: Presence-only, like `derive_meta_provenance`: a section that legitimately
#: measured nothing still produces its keys.
#:
#: `identity` is here because it is a *precondition*, not a result. Without a
#: named, hashed checkpoint the remaining numbers describe an unknown model, so
#: withholding `live` on its absence is the point.
REQUIRED_KEYS: Dict[str, Tuple[str, ...]] = {
    "identity": ("model_id", "revision", "weights_sha256", "config_sha256",
                 "tokenizer_sha256", "parameter_count", "n_layers", "n_heads",
                 "attested", "environment", "protocol"),
    "architecture": ("load_status", "model_name", "n_layers", "n_heads",
                     "d_model", "vocab_size"),
    "sanity_checks": tuple(name for name, *_rest in SANITY_CHECKS),
    "ioi": ("templates", "n_templates", "clean_correct", "corrupted_flipped"),
    "logit_lens": ("prompt", "layers"),
    "head_sweep": ("prompt", "n_layers", "n_heads", "n_heads_expected",
                   "n_heads_swept", "baseline_clean_ld", "ranking", "roles",
                   "top_heads", "supporting_heads", "opposing_heads", "all_heads"),
    "layer_ablation": ("prompt", "layers"),
    "steering": ("positive_prompt", "negative_prompt", "layers", "sweep"),
    "inspection": ("top_neurons", "head_detail"),
}


def validate_section(section: str, payload: Any) -> Dict[str, Any]:
    """Return the JSON-native payload, or raise if it is not a measurement.

    Two checks, both of which have caught a real failure mode:
    the payload must be a dict carrying the section's required keys, and it must
    survive JSON conversion without a fallback encoder.
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"{section} returned {type(payload).__name__}, expected a dict")
    required = REQUIRED_KEYS.get(section, ())
    absent = [key for key in required if key not in payload]
    if absent:
        raise ValueError(
            f"{section} is missing required key(s): {', '.join(absent)}")
    return to_json_native(payload, section)


@dataclass
class SectionResult:
    """What happened to one capture, whether or not it produced data.

    `payload` holds the validated measurement and is deliberately absent from
    `as_dict`: it is already the top-level `data[section]` value, and inlining
    it would double the size of the artifact.
    """

    section: str
    label: str
    status: str = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    payload: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
        }
        if self.error_type is not None:
            out["error"] = {
                "type": self.error_type,
                "message": self.error_message,
                "traceback": self.error_traceback,
            }
        return out


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def run_section(section: str, label: str, fn: Callable[[], Any]) -> SectionResult:
    """Run one capture, absorbing any failure into a recorded result.

    Never raises. The whole point is that a failure here still leaves the
    process able to write an artifact, so the provenance derivation can report
    `unavailable` instead of there being nothing to report at all.
    """
    result = SectionResult(section=section, label=label, status="running")
    result.started_at = _now()
    start = time.perf_counter()
    print(f"  {label} ...", end="", flush=True)
    try:
        result.payload = validate_section(section, fn())
        result.status = "ok"
    except BaseException as exc:  # noqa: BLE001 - the whole point is to survive
        # BaseException, not Exception: a KeyboardInterrupt or SystemExit
        # mid-sweep should still produce an artifact saying what had run, which
        # is precisely the information needed to decide whether to resume.
        result.status = "failed"
        result.error_type = type(exc).__name__
        result.error_message = str(exc)
        result.error_traceback = traceback.format_exc(limit=8)
        print(f"\n  [failed] {section}: {result.error_type}: {result.error_message}")
        return result
    finally:
        result.finished_at = _now()
        result.duration_s = round(time.perf_counter() - start, 3)

    print(f" [{result.duration_s:6.2f}s] ok")
    return result


def write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Write `payload` to `path` via a temp file and a rename.

    A crash or a full disk mid-write previously left a truncated `capture.json`
    behind -- an artifact that parses as JSON, is missing sections, and is
    indistinguishable from a run that legitimately measured less. `os.replace`
    is atomic on Windows and POSIX, so the destination is either the previous
    artifact or the complete new one.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        # The temp file is the only thing this call may leave behind. Removing it
        # keeps a failed run from depositing `capture.json.tmp` next to a good
        # artifact, where it reads as a second, newer result.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Model level
# --------------------------------------------------------------------------
def capture_identity() -> Dict[str, Any]:
    """Establish which weights produced everything else, and check they qualify.

    This is a capture rather than a constant for two reasons. An artifact that
    merely *says* "gpt2 (124M)" has not checked anything: if the engine loaded a
    different checkpoint the sentence would still be there. And a result whose
    weights cannot be named is not evidence about a model, so this section is
    required -- without it the run cannot reach `provenance: "live"`.

    The architecture check at the end is why this is also where a wrong model is
    rejected. The IOI protocol and its sanity checks are GPT-2-specific, so
    running them against other weights would yield numbers shaped like this
    protocol's and labelled as them.

    Loading happens here rather than in `capture_architecture`. Identity runs
    first and is a precondition for everything else, so it cannot assume a
    loaded model -- and if the load fails, there is no artifact claiming a model
    that is not there. That ordering was found by running it: the first attempt
    at this change reported "no model loaded" and exited 1, which is the
    behaviour working.
    """
    loaded = engine.load()
    if loaded.get("status") != "loaded":
        raise RuntimeError(
            f"weights did not load, so nothing here can be attributed to a "
            f"model: {loaded.get('status')!r} {loaded.get('error')!r}")

    identity = engine.model_identity()
    if not identity.get("attested"):
        raise RuntimeError(
            "model identity could not be established, so no result here can be "
            "attributed to specific weights: "
            f"{identity.get('error') or identity.get('attestation_reason')}")

    assert_protocol_applicable(identity)

    return {
        **identity,
        "environment": capture_environment(),
        "protocol": declared_protocol(),
    }


def assert_protocol_applicable(identity: Dict[str, Any]) -> None:
    """Raise unless the loaded weights match what this protocol assumes.

    Deliberately a rejection rather than an adjustment. Sizing the experiments to
    whatever is loaded is defensible for a *survey*; these are the published IOI
    and logit-lens numbers for GPT-2 small, and re-parameterising them under the
    same names would make the artifact claim something the protocol never
    measured.
    """
    mismatches = [
        f"{field}: loaded {identity.get(field)!r}, protocol expects {expected!r}"
        for field, expected in sorted(PROTOCOL_REFERENCE_ARCHITECTURE.items())
        if identity.get(field) != expected
    ]
    if mismatches:
        expected = ", ".join(f"{k}={v}" for k, v
                             in sorted(PROTOCOL_REFERENCE_ARCHITECTURE.items()))
        raise ValueError(
            "loaded weights do not match this experiment's protocol. "
            + "; ".join(mismatches)
            + f". The protocol is specified for {expected}; the loaded "
              f"checkpoint is {identity.get('revision') or 'unknown'}."
        )


def declared_protocol() -> Dict[str, Any]:
    """The experiment's inputs and selection rules, resolved and hashed.

    Recorded so a reader can separate what was chosen from what was measured. The
    hash covers every declared input, so a later edit to a prompt surfaces as a
    changed digest rather than as a silently different number.
    """
    dims = model_dims()
    neuron_layer = _resolve_fractions(
        (INSPECTION_NEURON_LAYER_FRACTION,), dims["n_layers"],
        "inspection neuron layer")[0]
    head_layer = _resolve_fractions(
        (INSPECTION_HEAD_LAYER_FRACTION,), dims["n_layers"],
        "inspection head layer")[0]
    head = _resolve_fractions(
        (INSPECTION_HEAD_FRACTION,), dims["n_heads"], "inspection head")[0]
    inputs = {
        "ioi_templates": [list(t) for t in IOI_TEMPLATES],
        "capital_prompt": CAPITAL_PROMPT,
        "steer_positive": STEER_POS,
        "steer_negative": STEER_NEG,
        "steer_alphas": list(STEER_ALPHAS),
        "sanity_checks": [list(c) for c in SANITY_CHECKS],
    }
    return {
        "inputs": inputs,
        "inputs_sha256": sha256_of(inputs),
        "selection": {
            "steering_layers": list(steering_layers(dims["n_layers"])),
            "steering_layer_fractions": list(STEERING_LAYER_FRACTIONS),
            "inspection_neuron_layer": neuron_layer,
            "inspection_head_layer": head_layer,
            "inspection_head": head,
            "inspection_fractions": {
                "neuron_layer": INSPECTION_NEURON_LAYER_FRACTION,
                "head_layer": INSPECTION_HEAD_LAYER_FRACTION,
                "head": INSPECTION_HEAD_FRACTION,
            },
            "note": (
                "Resolved from the loaded architecture. Which layers to use is a "
                "choice; the indices they resolve to are properties of this "
                "checkpoint, so they are recorded here rather than assumed."),
        },
        "reference_architecture": dict(PROTOCOL_REFERENCE_ARCHITECTURE),
    }


def sha256_of(value: Any) -> str:
    """Stable digest of a JSON-serialisable value."""
    payload = json.dumps(value, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nested(source: Any, *keys: str) -> Any:
    """Walk nested dicts, returning None rather than raising.

    `validate_section` checks that required keys are *present*, not that the
    values are the right type -- a stub or a future capture could satisfy it with
    a string where a dict belongs. The summary line in `meta` is not worth a
    crash over, so a miss degrades to None.
    """
    current = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _git(*args: str) -> Optional[str]:
    """Run a git command in the repo, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def capture_environment() -> Dict[str, Any]:
    """Everything outside the weights that changes the numbers.

    Recorded because a measurement without it can be neither reproduced nor
    explained: the same weights under a different torch or transformers release,
    run from a different revision of this script, can produce different logits.
    """
    versions: Dict[str, Any] = {"python": platform.python_version()}
    for module_name in ("torch", "numpy", "transformers"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except Exception as exc:  # noqa: BLE001
            versions[module_name] = f"unavailable: {exc}"

    try:
        with open(__file__, "rb") as handle:
            script_sha = "sha256:" + hashlib.sha256(handle.read()).hexdigest()
    except OSError as exc:
        script_sha = f"unavailable: {exc}"

    return {
        "versions": versions,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        # A commit alone is not enough: uncommitted changes are exactly the ones
        # that make a result unreproducible from the recorded hash.
        "git_dirty": bool(_git("status", "--porcelain")),
        "script": os.path.relpath(__file__, ROOT).replace("\\", "/"),
        "script_sha256": script_sha,
        "sampling": (
            "none; every capture is a deterministic forward pass with argmax "
            "decoding, so no random seed affects these numbers"),
        "torch_default_seed": torch.initial_seed(),
    }


def model_dims() -> Dict[str, int]:
    """Shape of the loaded model, read from the engine rather than assumed.

    Captures take their dimensions from here rather than from a literal, so a
    sweep covers every head the model actually has.
    """
    info = engine.info()
    keys = ("n_layers", "n_heads", "d_model", "d_head", "d_mlp", "vocab_size",
            "n_positions")
    missing = [k for k in keys if not isinstance(info.get(k), int)]
    if missing:
        raise RuntimeError(
            f"engine did not report {missing}; status={info.get('status')!r} "
            f"error={info.get('error')!r}")
    return {k: info[k] for k in keys}


def capture_architecture() -> Dict[str, Any]:
    load = engine.load()
    arch = engine.architecture()
    return {
        "load_status": load.get("status"),
        "model_name": arch.get("model_name"),
        "n_layers": arch.get("n_layers"),
        "n_heads": arch.get("n_heads"),
        "d_model": arch.get("d_model"),
        "d_mlp": arch.get("d_mlp"),
        "d_head": arch.get("d_head"),
        "vocab_size": arch.get("vocab_size"),
        "n_positions": arch.get("n_positions"),
        "n_params": arch.get("n_params"),
        "n_params_human": arch.get("n_params_human"),
        "device": arch.get("device"),
        "dtype": arch.get("dtype"),
    }


# --------------------------------------------------------------------------
# Sanity: prove the engine reproduces real GPT-2 behaviour
# --------------------------------------------------------------------------
def capture_sanity() -> Dict[str, Any]:
    """GPT-2 small behaviours that are unambiguous, used to prove the engine is
    not producing noise. These are the checks a reviewer should reproduce.

    Driven from `SANITY_CHECKS` so the prompt, the expected token and the prose
    description live in one declared table rather than being restated inline --
    and so the declared table's hash covers them. The expectation is reported
    with a `passes` flag, never asserted: a checkpoint that disagrees is a
    finding, not a reason to hide the section.
    """
    def top(prompt: str, k: int = 5) -> List[Dict[str, Any]]:
        run = engine.run_prompt(prompt)
        return run.get("top5", [])[:k]

    checks = {}
    for name, prompt, expected_token, expectation in SANITY_CHECKS:
        observed = top(prompt)
        checks[name] = {
            "prompt": prompt,
            "top": observed,
            "expected_top_token": expected_token,
            "expectation": expectation,
            "observed_top_token": (
                observed[0]["token"] if observed else None),
            "passes": bool(
                observed and observed[0]["token"].strip() == expected_token),
        }
    return checks


# --------------------------------------------------------------------------
# IOI: clean vs corrupted
# --------------------------------------------------------------------------
def capture_ioi() -> Dict[str, Any]:
    rows = []
    for prompt, correct, incorrect in IOI_TEMPLATES:
        engine.run_prompt(prompt)  # populate cache
        res = engine.ioi(correct.strip(), incorrect.strip())
        rows.append(
            {
                "prompt": prompt,
                "correct": correct,
                "incorrect": incorrect,
                "clean_top1": res.get("clean_top1"),
                "corrupted_top1": res.get("corrupted_top1"),
                "ioi_pass": res.get("ioi_pass"),
                "corrupted_pass": res.get("corrupted_pass"),
            }
        )
    n_pass = sum(1 for r in rows if r.get("ioi_pass"))
    n_flip = sum(1 for r in rows if r.get("corrupted_pass"))
    return {
        "templates": rows,
        "n_templates": len(rows),
        "clean_correct": n_pass,
        "corrupted_flipped": n_flip,
        "clean_accuracy": round(n_pass / len(rows), 4) if rows else None,
        "corrupted_flip_rate": round(n_flip / len(rows), 4) if rows else None,
    }


# --------------------------------------------------------------------------
# Logit lens
# --------------------------------------------------------------------------
def capture_logit_lens() -> Dict[str, Any]:
    lens = engine.logit_lens_all(CAPITAL_PROMPT, top_k=5)
    rows = []
    for entry in lens.get("layers", []):
        tops = entry.get("top_k_tokens") or []
        rows.append(
            {
                "layer": entry.get("layer"),
                "top_token": entry.get("top_token"),
                "top": [
                    {"token": t.get("token"), "prob": t.get("prob")}
                    if isinstance(t, dict)
                    else t
                    for t in (tops if isinstance(tops, list) else [])
                ],
            }
        )
    paris_at = next(
        (r["layer"] for r in rows if r["top_token"] == " Paris"),
        None,
    )
    paris_ranks = {
        r["layer"]: next(
            (
                i
                for i, t in enumerate(r["top"], start=1)
                if isinstance(t, dict) and t.get("token") == " Paris"
            ),
            None,
        )
        for r in rows
    }
    return {
        "prompt": CAPITAL_PROMPT,
        "layers": rows,
        "paris_first_argmax_layer": paris_at,
        "paris_rank_by_layer": paris_ranks,
    }


# --------------------------------------------------------------------------
# Full head zero-ablation sweep -> IOI circuit
# --------------------------------------------------------------------------
#: Roles a head can play under zero-ablation, by the sign of `effect`.
#:
#: A head whose removal makes the IOI logit difference *larger* is not part of
#: the IOI circuit -- it is working against it. Ranking by magnitude alone puts
#: such heads at the top of the list, which is how `L0H7`, `L0H0` and `L2H3`
#: came to head a table titled "which heads the IOI behaviour depends on" while
#: all three had *improved* IOI when ablated.
ROLE_SUPPORTS = "supports_ioi"
ROLE_OPPOSES = "opposes_ioi"
ROLE_NEUTRAL = "neutral"


def classify_head_effect(effect: float) -> str:
    """What a head's ablation did to the IOI signal, by sign alone.

    Strict sign, no threshold. The engine rounds its logit differences to four
    decimals, so a head whose true effect is below that resolution arrives as
    exactly `0.0` and is reported `neutral` rather than being given an invented
    boundary. A cutoff here would be one more number nobody chose deliberately.
    """
    if effect > 0:
        return ROLE_SUPPORTS
    if effect < 0:
        return ROLE_OPPOSES
    return ROLE_NEUTRAL


def _head_entry(layer: int, head: int, res: Dict[str, Any]) -> Dict[str, Any]:
    """One ablated head, with its signed causal effect spelled out.

    Three quantities that the old code conflated into one magnitude:

    * `delta` -- the engine's raw `patched_ld - clean_ld`.
    * `effect` -- `clean_ld - patched_ld`, the *signed* causal effect of the
      ablation on the IOI signal. Positive means removing the head destroyed
      the behaviour, which is what "this head is in the circuit" means. This is
      the quantity the ranking uses.
    * `effect_abs` -- magnitude, reported for readability but deliberately not
      the ranking key.

    `effect_fraction` normalises against the clean baseline so heads measured at
    different baseline strengths are comparable. It is signed for the same
    reason `effect` is, and is `None` when the clean difference is zero, where a
    ratio would be undefined rather than infinite.
    """
    clean_ld = res.get("clean_ld")
    patched_ld = res.get("patched_ld")
    delta = res.get("delta")

    effect = None
    if isinstance(clean_ld, (int, float)) and isinstance(patched_ld, (int, float)):
        effect = round(clean_ld - patched_ld, 4)

    # The sign convention is the whole point of this function, so it is checked
    # rather than assumed. If the engine ever flips `delta`, this fails loudly
    # instead of silently re-ranking every head.
    if effect is not None and isinstance(delta, (int, float)):
        if abs(effect - (-delta)) > 1e-3:
            raise ValueError(
                f"L{layer}H{head}: effect {effect} does not match -delta "
                f"{-delta}; the engine's delta convention has changed "
                f"(clean={clean_ld}, patched={patched_ld}, delta={delta})")

    fraction = None
    if effect is not None and isinstance(clean_ld, (int, float)) and clean_ld:
        fraction = round(effect / clean_ld, 4)

    return {
        "layer": layer,
        "head": head,
        "label": f"L{layer}H{head}",
        "clean_ld": clean_ld,
        "patched_ld": patched_ld,
        "delta": delta,
        "effect": effect,
        "effect_abs": abs(effect) if effect is not None else None,
        "effect_fraction": fraction,
        "role": classify_head_effect(effect) if effect is not None else ROLE_NEUTRAL,
    }


def _by_effect_desc(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Most destructive first, protective heads last.

    Tie-break on label so the ordering is total and reproducible: two heads with
    the same effect would otherwise swap places between runs, and an artifact
    whose ordering is unstable cannot be diffed against the previous one.
    """
    return sorted(entries, key=lambda h: (-(h["effect"] or 0.0), h["label"]))


def _by_effect_asc(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strongest opposition first.

    `opposing_heads` is read as "these heads work against the behaviour", so it
    leads with the one doing the most damage. Slicing it out of the descending
    ranking instead would put the *least* opposing head at the top of a list
    whose whole purpose is to say which opposition is strongest.
    """
    return sorted(entries, key=lambda h: ((h["effect"] or 0.0), h["label"]))


def capture_head_sweep() -> Dict[str, Any]:
    """Zero-ablate every head the model has and rank by the signed causal effect
    on the IOI logit difference. This is the standard first pass of IOI circuit
    discovery and it is fully measured, not simulated.

    The ranking used to be `abs(delta)`, which ranked a head that *supports* the
    IOI behaviour identically to one that *opposes* it. On the measured data
    that put three suppressing heads at the top of the circuit list. Ranking is
    now on `effect = clean_ld - patched_ld`, descending, so the heads whose
    removal destroys the behaviour come first and the ones whose removal helps
    come last.

    Dimensions come from `model_dims()`, and the sweep is checked for
    completeness: `patch_head` returns a non-ok status rather than raising when
    a head cannot be ablated, and the old loop `continue`d past those, so a
    partial sweep reported `n_heads_swept` without ever saying it was short.
    """
    dims = model_dims()
    n_layers, n_heads = dims["n_layers"], dims["n_heads"]
    prompt, correct, incorrect = IOI_TEMPLATES[0]
    engine.run_prompt(prompt)
    pos, neg = correct.strip(), incorrect.strip()

    heads: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    for layer in range(n_layers):
        for head in range(n_heads):
            res = engine.patch_head(layer, head, pos, neg)
            if res.get("status") != "ok":
                refused.append({
                    "layer": layer, "head": head,
                    "label": f"L{layer}H{head}",
                    "status": res.get("status"), "error": res.get("error"),
                })
                continue
            heads.append(_head_entry(layer, head, res))

    expected = n_layers * n_heads
    if len(heads) != expected:
        # An incomplete sweep is not a smaller result, it is a different one:
        # the ranking below is over whatever survived, and reporting it without
        # this would present a partial screen as a complete circuit search.
        raise RuntimeError(
            f"ablated {len(heads)} of {expected} heads ({n_layers} layers x "
            f"{n_heads} heads); {len(refused)} were refused, first: "
            f"{refused[0] if refused else None}. A partial sweep cannot be "
            f"ranked as a complete one.")

    ranked = _by_effect_desc(heads)
    supporting = _by_effect_desc([h for h in heads if h["role"] == ROLE_SUPPORTS])
    opposing = _by_effect_asc([h for h in heads if h["role"] == ROLE_OPPOSES])
    neutral = [h for h in ranked if h["role"] == ROLE_NEUTRAL]

    return {
        "prompt": prompt,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_heads_expected": expected,
        "n_heads_swept": len(heads),
        "baseline_clean_ld": next((h["clean_ld"] for h in heads), None),
        "ranking": {
            "key": "effect",
            "definition": "effect = clean_ld - patched_ld (signed)",
            "order": "descending; destructive ablations first, opposing heads last",
            "rationale": (
                "Ranked on the signed causal effect of the ablation, not on its "
                "magnitude. A head whose removal *increases* the IOI logit "
                "difference is working against the behaviour, and ranking it "
                "alongside heads whose removal destroys it conflated the two."),
        },
        "roles": {
            ROLE_SUPPORTS: len(supporting),
            ROLE_OPPOSES: len(opposing),
            ROLE_NEUTRAL: len(neutral),
        },
        "top_heads": ranked[:12],
        "supporting_heads": supporting,
        "opposing_heads": opposing,
        "all_heads": ranked,
    }


# --------------------------------------------------------------------------
# Layer ablation
# --------------------------------------------------------------------------
def capture_layer_ablation() -> Dict[str, Any]:
    dims = model_dims()
    prompt, correct, incorrect = IOI_TEMPLATES[0]
    rows = []
    for layer in range(dims["n_layers"]):
        res = engine.ablate_layer(layer, prompt, correct.strip(), incorrect.strip())
        rows.append(
            {
                "layer": layer,
                "clean_ld": res.get("clean_ld"),
                "patched_ld": res.get("patched_ld"),
                "delta": res.get("delta"),
                "direction": res.get("direction"),
            }
        )
    return {"prompt": prompt, "layers": rows}


# --------------------------------------------------------------------------
# Activation steering
# --------------------------------------------------------------------------
def capture_steering() -> Dict[str, Any]:
    """Contrast-vector steering: the steering vector is the measured difference
    between the residual stream on a positive and a negative prompt. Sweep the
    coefficient to find where the prediction actually flips.

    The layers were `(6, 8, 10)` -- a literal index list that described one
    checkpoint. They now come from `steering_layers(n_layers)`, which resolves
    to the same three layers here and records them, so the figure and the data
    cannot disagree about what was steered.
    """
    dims = model_dims()
    layers = steering_layers(dims["n_layers"])
    rows = []
    for layer in layers:
        for alpha in STEER_ALPHAS:
            res = engine.steer(CAPITAL_PROMPT, layer, STEER_POS, STEER_NEG, alpha)
            top5 = res.get("steered_top5") or []
            rows.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "vector_norm": res.get("vector_norm"),
                    "clean_top": res.get("clean_top"),
                    "steered_top": res.get("steered_top"),
                    "flipped": res.get("flipped"),
                    "paris_rank": next(
                        (i for i, t in enumerate(top5, start=1) if t == " Paris"), None
                    ),
                    "steered_top5": top5,
                }
            )
    return {
        "positive_prompt": STEER_POS,
        "negative_prompt": STEER_NEG,
        "prompt": CAPITAL_PROMPT,
        "layers": list(layers),
        "alphas": list(STEER_ALPHAS),
        "n_layers_swept": dims["n_layers"],
        "sweep": rows,
        "flips": [r for r in rows if r.get("flipped")],
    }


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------
def capture_inspection() -> Dict[str, Any]:
    """Inspect one MLP layer and one attention head, both resolved from the model.

    This was `list_neurons(8, ...)` and `head_detail(9, 9)`, with the indices
    baked into the *result keys* as well -- `top_neurons_L8_by_in_norm` and
    `head_L9H9`. A key naming a layer the run never visited is worse than a
    stale number: it survives the mistake. The keys are now dimension-free and
    the indices live in the payload.
    """
    dims = model_dims()
    neuron_layer, head_layer, head_index = inspection_indices(
        dims["n_layers"], dims["n_heads"])
    neurons = engine.list_neurons(neuron_layer, "mlp", 0, 10, "in_norm", "desc")
    head = engine.head_detail(head_layer, head_index)
    return {
        "top_neurons": {
            "layer": neuron_layer,
            "component": "mlp",
            "rank_by": "in_norm",
            "order": "desc",
            "neurons": (neurons.get("neurons") or [])[:10],
        },
        "head_detail": {
            "layer": head_layer,
            "head": head_index,
            "label": f"L{head_layer}H{head_index}",
            "detail": head,
        },
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def render_figures(data: Dict[str, Any]) -> List[str]:
    """Render figures from whatever sections produced validated data.

    Each figure reads its section through `_section`, so a failed section
    yields no figure instead of a `KeyError` that would abort the run *after*
    the captures had finished -- losing the artifact the run just earned. A
    figure missing because its data is missing is also the honest outcome: a
    plot of nothing is not evidence of nothing.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"  [skip figures] matplotlib unavailable: {exc}")
        return []

    os.makedirs(IMG_DIR, exist_ok=True)
    written = []

    def _section(name: str) -> Optional[Dict[str, Any]]:
        value = data.get(name)
        return value if isinstance(value, dict) else None

    # 1. IOI head sweep
    sweep = _section("head_sweep")
    heads = (sweep or {}).get("all_heads") or []
    grid = np.full((
        max((h["layer"] for h in heads), default=-1) + 1,
        max((h["head"] for h in heads), default=-1) + 1,
    ), np.nan)
    for h in heads:
        effect = h.get("effect")
        if isinstance(effect, float):
            grid[h["layer"], h["head"]] = effect

    # Nothing plottable is no figure, rather than a warning and an empty grid.
    # `np.nanmax` over an all-NaN slice warns and returns nan, which would then
    # propagate into the colour limits.
    if np.isfinite(grid).any():
        n_layers, n_heads = grid.shape
        limit = float(np.nanmax(np.abs(grid))) or 1.0
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit,
                       aspect="auto")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title("IOI zero-ablation: signed effect on logit difference per head\n"
                     "warm = removal destroys IOI (supports); cool = removal helps IOI")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        fig.colorbar(im, ax=ax, label="effect (clean − ablated)")
        fig.tight_layout()
        path = os.path.join(IMG_DIR, "ioi-head-sweep.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    # 2. Layer ablation
    ablation = _section("layer_ablation")
    rows = (ablation or {}).get("layers") or []
    if rows:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.bar(
            [r["layer"] for r in rows],
            [r["delta"] for r in rows],
            color="#2563eb",
        )
        ax.axhline(0, color="#334155", lw=0.8)
        ax.set_xlabel("Ablated layer")
        ax.set_ylabel("Δ IOI logit difference")
        ax.set_title("Leave-one-layer-out ablation on the IOI prompt")
        ax.set_xticks(range(len(rows)))
        fig.tight_layout()
        path = os.path.join(IMG_DIR, "ioi-layer-ablation.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    # 3. Logit lens trajectory
    lens = _section("logit_lens")
    rows = (lens or {}).get("layers") or []
    if rows:
        layers = [row["layer"] for row in rows]
        probs = [
            next(
                (
                    t.get("prob")
                    for t in row["top"]
                    if isinstance(t, dict) and t.get("token") == " Paris"
                ),
                0.0,
            )
            or 0.0
            for row in rows
        ]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(layers, probs, marker="o", color="#2563eb", lw=2)
        ax.fill_between(layers, probs, color="#2563eb", alpha=0.12)
        ax.set_xlabel("Layer")
        ax.set_ylabel('P(" Paris") at that layer')
        ax.set_title('Logit lens: probability of " Paris" per layer')
        ax.set_xticks(layers)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = os.path.join(IMG_DIR, "logit-lens-paris.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    # 4. Steering sweep: rank of " Paris" as steering strength increases
    steering = _section("steering")
    rows = (steering or {}).get("sweep") or []
    if rows:
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        for layer in sorted({r["layer"] for r in rows}):
            sel = sorted(
                (r for r in rows if r["layer"] == layer), key=lambda r: r["alpha"]
            )
            ax.plot(
                [r["alpha"] for r in sel],
                [r["paris_rank"] if r["paris_rank"] else 6 for r in sel],
                marker="o",
                label=f"layer {layer}",
            )
        ax.axhline(1, color="#13795f", ls="--", lw=1.2)
        ax.text(
            5.5,
            1.15,
            "rank 1 = steering wins",
            color="#13795f",
            fontsize=9,
        )
        ax.set_yticks([1, 2, 3, 4, 5, 6])
        ax.set_yticklabels(["1", "2", "3", "4", "5", "not in top 5"])
        ax.invert_yaxis()
        ax.set_xlabel("Steering coefficient α")
        ax.set_ylabel('Rank of " Paris" in steered top-5')
        ax.set_title("Contrast-vector steering: France prompt toward Paris")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = os.path.join(IMG_DIR, "steering-sweep.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    return written


#: The captures this script performs, in order: (section key, console label,
#: capture function *name*). A single registry rather than eight statements in
#: `main()`, so `EXPECTED_SECTIONS` describes the code instead of restating it,
#: and adding a capture is one edit.
#:
#: The function is referenced by name, not held as an object, and resolved in
#: `run_captures`. A registry holding live references looks equivalent and is
#: not: `monkeypatch.setattr(capture_results, "capture_ioi", boom)` rebinds the
#: module attribute, while the registry would keep calling the original and the
#: injected failure would silently not happen. In a change whose entire subject
#: is "a failing capture must not pass unnoticed", the one handle a reader would
#: reach for to make a capture fail has to work.
CAPTURES: Tuple[Tuple[str, str, str], ...] = (
    # First, and required: nothing below it can be attributed to a model until
    # this establishes which one, and a run whose identity section failed cannot
    # reach `provenance: "live"`.
    ("identity", "model identity + environment", "capture_identity"),
    ("architecture", "architecture", "capture_architecture"),
    ("sanity_checks", "sanity checks", "capture_sanity"),
    ("ioi", "ioi (3 templates)", "capture_ioi"),
    ("logit_lens", "logit lens", "capture_logit_lens"),
    ("head_sweep", "full head zero-ablation sweep", "capture_head_sweep"),
    ("layer_ablation", "layer ablation", "capture_layer_ablation"),
    ("steering", "steering sweep", "capture_steering"),
    ("inspection", "inspection", "capture_inspection"),
)

#: The claim that a complete run covers all of them. Deliberately written out
#: rather than derived from `CAPTURES`: if it were derived, the test comparing
#: the two would compare a tuple with itself and pass vacuously. A new capture
#: that is not listed here does not block the run -- it is simply not claimed --
#: which is the safe direction to fail in.
EXPECTED_SECTIONS = (
    "identity", "architecture", "sanity_checks", "ioi", "logit_lens",
    "head_sweep", "layer_ablation", "steering", "inspection",
)


def derive_meta_provenance(data: Dict[str, Any],
                           results: Optional[List[SectionResult]] = None) -> Dict[str, Any]:
    """Report which sections actually produced data, and withhold `live` if not.

    This used to be a literal `"provenance": "live"` in the `meta` dict at the
    top of `main()`, stamped before a single forward pass had run and never
    revised -- `provenance` appeared exactly once in this file, in that literal.
    Every capture below it is a genuine measurement, but the flag asserted the
    outcome instead of deriving it, so any future change that let a capture fail
    quietly would keep writing `live` into the artifact the README's Results
    section is generated from.

    Presence, not truthiness: a section that legitimately found nothing -- an
    empty head sweep is a real result, not a missing measurement -- still ran and
    still measured. Only a key absent from `data` did not run.

    `results` carries the per-section run report when the caller has one. It is
    used only to name the *cause* of an absent section; the verdict itself is
    derived from `data`, so this function stays honest when called with nothing
    but the data it is given.
    """
    measured = [k for k in EXPECTED_SECTIONS if k in data]
    missing = [k for k in EXPECTED_SECTIONS if k not in data]

    out: Dict[str, Any] = {
        "measured_sections": f"{len(measured)}/{len(EXPECTED_SECTIONS)}",
        "provenance": "live" if not missing else "unavailable",
    }
    if missing:
        failures = {r.section: r for r in (results or []) if r.status != "ok"}
        detail = []
        for key in missing:
            result = failures.get(key)
            if result is None:
                detail.append(f"{key} (no run report)")
            elif result.error_type:
                detail.append(f"{key} ({result.error_type}: {result.error_message})")
            else:
                detail.append(f"{key} ({result.status})")
        out["reason"] = (
            "No data for: " + ", ".join(detail)
            + ". The artifact is incomplete, so it does not describe a full "
              "measurement run and must not be cited as one."
        )
        out["note"] = (
            "INCOMPLETE RUN. Some sections did not produce data, so this file "
            "does not claim that every value was measured."
        )
    return out


def run_captures() -> Tuple[Dict[str, Any], List[SectionResult]]:
    """Run every registered capture, returning the data and the run report.

    Each capture is isolated: one failing cannot prevent the others from
    producing data, and cannot prevent this function from returning.
    """
    data: Dict[str, Any] = {}
    results: List[SectionResult] = []
    for section, label, fn_name in CAPTURES:
        try:
            fn = globals()[fn_name]
        except KeyError:
            result = SectionResult(section=section, label=label, status="failed")
            result.error_type = "NameError"
            result.error_message = f"no capture named {fn_name!r} in this module"
            results.append(result)
            print(f"  [failed] {section}: {result.error_type}: {result.error_message}")
            continue

        result = run_section(section, label, fn)
        results.append(result)
        if result.status != "ok":
            continue
        # The payload is carried on the result rather than re-running the capture
        # to fetch it: a capture is a forward pass, and re-running would double
        # the cost and could produce a different value than the recorded timings
        # and status describe.
        data[section] = result.payload
    return data, results


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    started_at = _now()
    print("Establishing model identity ...")

    captured, results = run_captures()

    data: Dict[str, Any] = {"meta": {
        "generated_by": "scripts/capture_results.py",
        "run_started_at": started_at,
        # provenance is set after the captures run, not asserted here -- see the
        # derivation below. It used to sit in this literal, stamped "live" before
        # a single forward pass had happened and never revised.
        "note": (
            "Every value is measured from live GPT-2 weights by a real forward "
            "pass. No synthetic, sampled or literature-copied values."),
    }}
    # Only validated measurements reach `data`. A section that failed or failed
    # validation is represented in `sections` below and absent here, which is
    # what makes the presence-based derivation withhold `live`.
    data.update(captured)

    # The model line is read out of the identity capture, never written here.
    # It used to be the literal `"gpt2 (124M)"`, which described weights nobody
    # had checked: a different checkpoint loaded under that sentence would have
    # produced the same artifact. When identity is absent the line says so
    # instead of guessing, because the whole point of the section is that it
    # cannot be assumed.
    identity = captured.get("identity")
    if isinstance(identity, dict):
        data["meta"]["model"] = (
            f"{identity.get('model_id')} "
            f"({identity.get('parameter_count'):,} parameters, "
            f"revision {identity.get('revision') or 'unknown'})"
            if isinstance(identity.get("parameter_count"), int)
            else str(identity.get("model_id"))
        )
        data["meta"]["weights_sha256"] = identity.get("weights_sha256")
        torch_version = _nested(identity, "environment", "versions", "torch")
        data["meta"]["library"] = (
            f"transformers + torch {torch_version}" if torch_version
            else "transformers + torch"
        )
        # Pointer, not a copy: the authoritative block is `data["identity"]`.
        data["meta"]["model_identity"] = "data['identity']"
    else:
        data["meta"]["model"] = "unidentified"
        data["meta"]["library"] = "unidentified"

    figures = render_figures(data)

    data["figures"] = [os.path.relpath(p, ROOT).replace("\\", "/") for p in figures]
    data["sections"] = {r.section: r.as_dict() for r in results}

    data["meta"].update(derive_meta_provenance(data, results))
    if data["meta"]["provenance"] != "live":
        print(f"  [incomplete] {data['meta']['reason']}")

    path = os.path.join(OUT_DIR, "capture.json")
    write_json_atomic(path, data)

    print(f"\nWrote {os.path.relpath(path, ROOT)}")
    if isinstance(identity, dict):
        print(f"  model: {identity.get('model_id')} "
              f"revision {identity.get('revision')}")
        print(f"  weights: {identity.get('weights_sha256')}")
    for fig in data["figures"]:
        print(f"  figure: {fig}")
    for result in results:
        if result.status != "ok":
            print(f"  FAILED {result.section}: "
                  f"{result.error_type}: {result.error_message}")

    # A non-zero exit on an incomplete run: `capture.json` exists and says
    # `unavailable`, so a CI step or a reviewer sees the failure without having
    # to open the file. Returning 0 here is what lets a broken capture look like
    # a successful one to anything downstream of this script.
    return 0 if data["meta"]["provenance"] == "live" else 1


if __name__ == "__main__":
    raise SystemExit(main())
