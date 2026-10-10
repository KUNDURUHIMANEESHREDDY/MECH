"""No value in the capture artifact may be a literal.

The rule under test
-------------------
`scripts/capture_results.py` writes `docs/results/capture.json`, the artifact
the README's Results section quotes. It used to carry a literal

    "meta": {..., "model": "gpt2 (124M)", ...}

alongside hardcoded layer indices: `list_neurons(8, ...)`, `head_detail(9, 9)`,
steering layers `(6, 8, 10)`, a `12x12` sweep label, and result keys literally
named `top_neurons_L8_by_in_norm` and `head_L9H9`.

Three problems, in increasing order of how long they hide. The artifact
described weights nobody had checked. The inspected layers were baked into the
*keys*, so a checkpoint of a different depth would have been measured at the
same indices under a label claiming otherwise. And a reader had no way to tell a
measurement from a constant.

So identity became a required capture, layer choices became fractions of the
loaded depth resolved against the architecture, and the environment that changes
the numbers is recorded alongside them.

What is asserted here
---------------------
* No literal model name, parameter count or checkpoint id survives in code.
* `meta["model"]` is read from the identity capture, so a different engine
  produces a different line.
* An unattested identity cannot reach `provenance: "live"`.
* Weights that do not match the protocol's reference architecture are rejected
  rather than measured and labelled.
* Selection indices track the loaded architecture -- and at GPT-2 small they
  resolve to exactly the values that were hardcoded, so the published numbers
  are unchanged.
* Result keys carry no layer digits.
* A refused head fails the sweep instead of quietly shrinking it.

Negative controls: restoring `"model": "gpt2 (124M)"`, hardcoding the steering
layers back to `(6, 8, 10)`, and reverting the architecture check to a warning
were each caught.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capture_results.py"
ENGINE = ROOT / "backend" / "services" / "gpt2_engine.py"

from scripts import capture_results as cap  # noqa: E402

LIVE = "live"
UNAVAILABLE = "unavailable"

GPT2_SMALL = {
    "n_layers": 12, "n_heads": 12, "d_model": 768, "d_head": 64,
    "d_mlp": 3072, "vocab_size": 50257, "n_positions": 1024,
}


# ── Fakes ───────────────────────────────────────────────────────────────

def _identity(**overrides) -> Dict[str, Any]:
    base = {
        "status": "ok",
        "attested": True,
        "model_id": "gpt2",
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "weights_sha256": "sha256:" + "1" * 64,
        "config_sha256": "sha256:" + "2" * 64,
        "tokenizer_sha256": "sha256:" + "3" * 64,
        "tokenizer_class": "GPT2TokenizerFast",
        "tokenizer_vocab_size": 50257,
        "parameter_count": 124439808,
        "dtype": "torch.float32",
        "device": "cpu",
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_head": 64,
        "d_mlp": 3072,
        "vocab_size": 50257,
        "n_positions": 1024,
        "activation": "gelu_new",
    }
    base.update(overrides)
    return base


class FakeEngine:
    """The slice of `gpt2_engine` the captures touch.

    Duck-typed on purpose, matching the precedent in
    `test_model_attestation.py`: the tests must not need 500MB of weights to
    check that an index is derived.
    """

    def __init__(self, dims=None, identity=None, steer_result=None,
                 refuse_heads=(), neuron_result=None, head_result=None,
                 ioi_result=None, lens=None, load_result=None):
        self.dims = dict(GPT2_SMALL if dims is None else dims)
        self.identity = identity if identity is not None else _identity()
        self.load_result = load_result or {"status": "loaded"}
        self.steer_calls: List[Dict[str, Any]] = []
        self.neuron_calls: List[Dict[str, Any]] = []
        self.head_calls: List[Dict[str, Any]] = []
        self.refuse_heads = set(refuse_heads)
        self._steer = steer_result or {"steered_top5": [" Paris"], "flipped": True}
        self._neurons = neuron_result or {"neurons": [{"idx": i} for i in range(30)]}
        self._head = head_result or {"status": "ok"}
        self._ioi = ioi_result or {
            "clean_top1": " Mary", "corrupted_top1": " John",
            "ioi_pass": True, "corrupted_pass": True,
        }
        self._lens = lens or {"layers": [{"layer": i, "top_token": " Paris",
                                           "top_k_tokens": []} for i in range(12)]}

    def info(self):
        return {"status": "loaded", "model_name": self.identity.get("model_id"),
                "n_params": self.identity.get("parameter_count"),
                "n_params_human": "124.4M", **self.dims}

    def load(self):
        return self.load_result

    def model_identity(self):
        return dict(self.identity)

    def architecture(self):
        return {"status": "ok", **self.dims}

    def run_prompt(self, prompt):
        return {"top5": [{"token": " Jul", "logit": -1.0, "prob": 0.9},
                         {"token": " Jun", "logit": -2.0, "prob": 0.05}]}

    def ioi(self, correct, incorrect):
        return dict(self._ioi)

    def logit_lens_all(self, prompt, top_k=5):
        return self._lens

    def patch_head(self, layer, head, pos, neg):
        if (layer, head) in self.refuse_heads:
            return {"status": "error", "error": "cannot ablate this head"}
        # `delta` must be patched - clean, matching the engine's convention; the
        # capture now checks that, so an inconsistent fake would fail its own test.
        return {"status": "ok", "clean_ld": 4.0, "patched_ld": 1.0, "delta": -3.0}

    def ablate_layer(self, layer, prompt, correct, incorrect):
        return {"clean_ld": 4.0, "patched_ld": 2.0, "delta": 2.0,
                "direction": "decrease"}

    def steer(self, prompt, layer, pos_prompt, neg_prompt, alpha):
        self.steer_calls.append({"prompt": prompt, "layer": layer, "alpha": alpha})
        return dict(self._steer)

    def list_neurons(self, layer, *args, **kwargs):
        self.neuron_calls.append({"layer": layer, "args": args, "kwargs": kwargs})
        return self._neurons

    def head_detail(self, layer, head):
        self.head_calls.append({"layer": layer, "head": head})
        return self._head


@pytest.fixture
def fake_engine(monkeypatch):
    def install(engine: FakeEngine) -> FakeEngine:
        monkeypatch.setattr(cap, "engine", engine)
        return engine
    return install


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cap, "OUT_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(cap, "IMG_DIR", str(tmp_path / "images"))
    monkeypatch.setattr(cap, "render_figures", lambda data: [])
    return tmp_path


def _artifact(tmp_path: Path) -> Dict[str, Any]:
    return json.loads((tmp_path / "results" / "capture.json").read_text(
        encoding="utf-8"))


def _identity_payload(**overrides) -> Dict[str, Any]:
    """An identity block that satisfies `REQUIRED_KEYS`, for stubbing."""
    base = _identity(**overrides)
    base.setdefault("environment", {"versions": {"torch": "2.14.0"}})
    base.setdefault("protocol", {"inputs_sha256": "sha256:" + "4" * 64})
    return base


def _stub_all(monkeypatch, identity=None):
    """Replace every capture with a valid stub, optionally with an identity."""
    payloads: Dict[str, Any] = {}
    for section in cap.EXPECTED_SECTIONS:
        if section == "identity":
            continue
        if section == "sanity_checks":
            payloads[section] = {
                name: {"prompt": "p", "top": [], "passes": False}
                for name, *_rest in cap.SANITY_CHECKS
            }
        else:
            payloads[section] = {k: "x" for k in cap.REQUIRED_KEYS[section]}

    for section, _label, name in cap.CAPTURES:
        if section == "identity":
            if identity is None:
                continue
            payloads[section] = identity
        payload = payloads[section]
        monkeypatch.setattr(cap, name, lambda payload=payload: dict(payload))


# ── No literals survive ─────────────────────────────────────────────────

def _docstring_free_strings(path: Path):
    """Every string literal in `path` except module/function docstrings.

    The prose in this file names the literals it removed -- `gpt2 (124M)`,
    `head_L9H9`, `(6, 8, 10)` -- and a grep would match those explanations as if
    they were the defect. This repository has been bitten by exactly that: see
    `test_no_hash_derived_identifiers`, whose docstring quotes the old code in
    order to explain why it was wrong.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue
            out.append((node.lineno, node.value))
    return out


def _dict_literal_values(path: Path):
    """`{key: value}` pairs written as literals, so a *description* can be told
    from a *request*.

    `"gpt2"` is a legitimate constant in `gpt2_engine` -- it is the repo id
    passed to `from_pretrained`, the thing being asked for. `"model_name": "gpt2"`
    is not: that is a claim about the weights in memory. Both are the same
    string, so the distinction has to be structural.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and isinstance(key.value, str)
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)):
                out.append((node.lineno, key.value, value.value))
    return out


def _reference_architecture_lines() -> set:
    """Lines of `PROTOCOL_REFERENCE_ARCHITECTURE`, located by AST.

    Located rather than declared: adding a line-number constant to the module so
    a test could reference it would be a test hook in production code, and it
    would go stale the moment anything above it moved.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and any(getattr(t, "id", None) == "PROTOCOL_REFERENCE_ARCHITECTURE"
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))
    )
    return {n.lineno for n in ast.walk(node) if hasattr(n, "lineno")}


def test_no_model_name_literal_remains_in_the_capture_script():
    """`"model": "gpt2 (124M)"` and friends.

    The one permitted mention is `PROTOCOL_REFERENCE_ARCHITECTURE["model_id"]`,
    which exists to be *compared against* the loaded model -- see
    `test_the_reference_architecture_is_a_check_not_a_source_of_values`. The
    allowance is by AST node, so a second mention would be a description
    sneaking back in.
    """
    allowed = _reference_architecture_lines()
    offenders = [
        f"line {line}: {value!r}"
        for line, value in _docstring_free_strings(SCRIPT)
        if "gpt2" in value.lower() and line not in allowed
    ]
    assert not offenders, (
        "a GPT-2 name is written as a literal again, so the artifact can describe "
        "weights that were never loaded:\n  " + "\n  ".join(offenders))


def test_no_checkpoint_id_or_parameter_count_literal_remains():
    offenders = [
        f"line {line}: {value!r}"
        for line, value in _docstring_free_strings(SCRIPT)
        if "124" in value
    ]
    assert not offenders, (
        "a parameter count is written as a literal again:\n  "
        + "\n  ".join(offenders))


def test_the_engine_no_longer_asserts_identity_in_a_response():
    """`info()` and `neuron_detail()` used to answer `"model_name": "gpt2"`.

    A request for a repo id may be a literal -- that is `from_pretrained`'s
    argument, the thing being asked for. A *response* describing weights in
    memory may not: it is the claim this whole change is about. Checked
    structurally, because both are the same six characters.
    """
    description_keys = {"model_name", "activation", "architecture", "model_type"}
    offenders = [
        f"line {line}: {key}={value!r}"
        for line, key, value in _dict_literal_values(ENGINE)
        if key in description_keys
    ]
    assert not offenders, (
        "gpt2_engine asserts a description of its weights instead of reading it "
        "off the loaded model:\n  " + "\n  ".join(offenders))


def test_the_engine_still_requests_a_concrete_checkpoint():
    """The request stays a literal.

    Worth stating so the assertion above cannot be 'fixed' by deleting the
    checkpoint: `from_pretrained` needs to be told what to load, and a request is
    not a claim.
    """
    source = ENGINE.read_text(encoding="utf-8")
    assert 'from_pretrained("gpt2")' in source
    assert 'SUPPORTED_MODEL = "gpt2"' in source


def test_model_identity_reports_an_error_rather_than_a_default_when_unloaded():
    """No weights means no identity. The old shape would have let a caller fall
    back to a name."""
    from backend.services import gpt2_engine

    original = gpt2_engine._model
    try:
        gpt2_engine._model = None
        identity = gpt2_engine.model_identity()
        info = gpt2_engine.info()
    finally:
        gpt2_engine._model = original

    assert identity["attested"] is False
    assert identity["status"] == "error"
    assert info["status"] == "error"


# ── Identity is required ────────────────────────────────────────────────

def test_identity_is_a_required_section():
    assert "identity" in cap.EXPECTED_SECTIONS
    assert "identity" in cap.REQUIRED_KEYS
    assert cap.CAPTURES[0][0] == "identity", (
        "identity must run first; nothing below it can be attributed to a "
        "model until it has been established")


def test_a_run_without_identity_cannot_report_live():
    """The defect this section exists for: an artifact announcing a model it
    never established."""
    data = {section: {"result": 1} for section in cap.EXPECTED_SECTIONS
            if section != "identity"}

    derived = cap.derive_meta_provenance(data)

    assert derived["provenance"] == UNAVAILABLE
    assert "identity" in derived["reason"]


def test_meta_model_comes_from_the_identity_capture(isolated, fake_engine,
                                                    monkeypatch):
    payload = _identity_payload(model_id="distilgpt2", parameter_count=81_912_576,
                                revision="deadbeef" * 5)
    fake_engine(FakeEngine(identity=payload))
    _stub_all(monkeypatch, identity=payload)

    cap.main()
    meta = _artifact(isolated)["meta"]

    assert "distilgpt2" in meta["model"]
    assert "81,912,576" in meta["model"]
    assert "deadbeef" in meta["model"]
    assert "gpt2 (124M)" not in json.dumps(meta)


def test_an_absent_identity_leaves_the_model_line_unidentified(isolated,
                                                             fake_engine,
                                                             monkeypatch):
    """Better a hole than a guess.

    The whole point of the section is that identity cannot be assumed, so when
    it fails the line says `unidentified` rather than falling back to the name
    that used to be hardcoded there.
    """
    fake_engine(FakeEngine())
    _stub_all(monkeypatch, identity=None)
    monkeypatch.setattr(
        cap, "capture_identity",
        lambda: (_ for _ in ()).throw(RuntimeError("no weights")))

    cap.main()
    meta = _artifact(isolated)["meta"]

    assert meta["model"] == "unidentified"
    assert meta["provenance"] == UNAVAILABLE


def test_an_unattested_identity_is_refused(isolated, fake_engine, monkeypatch):
    """The engine could not name the weights, so the run must not claim live.

    `capture_identity` itself runs here -- only the other captures are stubbed --
    because the point is that the refusal comes from the identity section rather
    than from whatever the test stubbed into it.
    """
    fake_engine(FakeEngine(identity={
        "status": "error", "attested": False,
        "error": "hashing failed: vocab unreadable"}))
    _stub_all(monkeypatch, identity=None)

    exit_code = cap.main()
    artifact = _artifact(isolated)

    assert exit_code != 0
    assert artifact["meta"]["provenance"] == UNAVAILABLE
    assert "identity" not in artifact
    error = artifact["sections"]["identity"]["error"]
    assert error["type"] == "RuntimeError"
    assert "hashing failed" in error["message"]


def test_identity_loads_the_weights_rather_than_assuming_them(fake_engine):
    """Identity runs first and is a precondition for everything else, so it
    cannot rely on a later capture having loaded the model.

    Found by running the change: the first version reported "no model loaded"
    and exited 1 -- the fail-closed path working, on a bug I had just written.
    """
    fake_engine(FakeEngine())
    identity = cap.capture_identity()
    assert identity["attested"] is True


def test_a_failed_load_stops_the_run_before_any_measurement(fake_engine):
    """No artifact may name a model that is not there.

    If the weights will not load, there is nothing to attribute a result to, so
    the refusal happens in the first section rather than after a 30-second sweep
    has produced numbers nobody can place.
    """
    fake_engine(FakeEngine(load_result={"status": "error",
                                        "error": "connection reset"}))

    with pytest.raises(RuntimeError) as excinfo:
        cap.capture_identity()

    assert "weights did not load" in str(excinfo.value)
    assert "connection reset" in str(excinfo.value)


def test_capture_identity_raises_when_nothing_can_be_hashed(fake_engine):
    fake_engine(FakeEngine(identity={
        "status": "error", "attested": False,
        "attestation_reason": "vocab cannot be hashed"}))

    with pytest.raises(RuntimeError) as excinfo:
        cap.capture_identity()

    assert "attributed to specific weights" in str(excinfo.value)
    assert "vocab cannot be hashed" in str(excinfo.value)


# ── Wrong-model rejection ───────────────────────────────────────────────

def test_mismatched_weights_are_rejected_rather_than_measured(fake_engine):
    fake_engine(FakeEngine(identity=_identity(n_layers=24, d_model=1024)))

    with pytest.raises(ValueError) as excinfo:
        cap.capture_identity()

    message = str(excinfo.value)
    assert "do not match this experiment's protocol" in message
    assert "n_layers" in message and "24" in message
    assert "d_model" in message


@pytest.mark.parametrize("field,value", [
    ("model_id", "distilgpt2"),
    ("n_heads", 16),
    ("d_model", 1024),
    ("vocab_size", 50257),
])
def test_each_protocol_field_is_actually_checked(field, value):
    """Every field, not a sample: a check that silently stopped comparing one
    of them is the same defect as no check."""
    identity = _identity(**{field: value})

    if identity == _identity():
        return
    with pytest.raises(ValueError) as excinfo:
        cap.assert_protocol_applicable(identity)
    assert field in str(excinfo.value)


def test_matching_weights_pass():
    cap.assert_protocol_applicable(_identity())


def test_the_reference_architecture_is_a_check_not_a_source_of_values():
    """It has to exist -- a protocol cannot be validated against nothing -- but
    it must not move with the model, or it is not a reference.

    A reference that followed the loaded checkpoint would accept every model,
    which is the opposite of a check.
    """
    assert set(cap.PROTOCOL_REFERENCE_ARCHITECTURE) == {
        "model_id", "n_layers", "n_heads", "d_model", "vocab_size"}
    assert cap.PROTOCOL_REFERENCE_ARCHITECTURE["n_layers"] == 12

    # It still says 12 when 24 layers are loaded, which is what makes the
    # mismatch in `assert_protocol_applicable` reachable at all.
    assert cap.PROTOCOL_REFERENCE_ARCHITECTURE["n_layers"] != 24


def test_a_reported_dimension_comes_from_the_weights_not_the_table(fake_engine):
    fake_engine(FakeEngine(dims={**GPT2_SMALL, "n_layers": 24}))

    assert cap.model_dims()["n_layers"] == 24
    assert cap.capture_head_sweep()["n_layers"] == 24


# ── Derived selection ───────────────────────────────────────────────────

def test_the_derived_indices_reproduce_the_published_ones():
    """At GPT-2 small these must equal what was hardcoded, or the published
    measurements would change.

    Steering was (6, 8, 10); the neuron inspection was `list_neurons(8, ...)` and
    the head inspection was `head_detail(9, 9)`. Note the head is on layer 9,
    not 8 -- an earlier version of this change resolved one layer for both and
    moved the head sweep to L8H9 while every other number stayed identical.
    """
    assert cap.steering_layers(12) == (6, 8, 10)
    assert cap.inspection_indices(12, 12) == (8, 9, 9)


def test_selection_tracks_a_different_architecture():
    assert cap.steering_layers(24) == (12, 16, 20)
    assert cap.inspection_indices(24, 16) == (16, 18, 12)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_depth_too_small_for_the_rule_is_refused_not_clamped(depth):
    """Clamping would report the same layer several times under a sweep's name,
    or measure layer 0 for a fraction that meant the middle."""
    with pytest.raises(ValueError):
        cap.steering_layers(depth)


def test_steering_sweeps_the_layers_the_model_actually_has(fake_engine):
    engine = FakeEngine(dims={**GPT2_SMALL, "n_layers": 24})
    fake_engine(engine)

    payload = cap.capture_steering()

    assert payload["layers"] == [12, 16, 20]
    assert {c["layer"] for c in engine.steer_calls} == {12, 16, 20}
    assert sorted({c["alpha"] for c in engine.steer_calls}) == sorted(
        cap.STEER_ALPHAS)


def test_inspection_visits_the_resolved_layers_and_head(fake_engine):
    engine = FakeEngine(dims={**GPT2_SMALL, "n_layers": 24, "n_heads": 16})
    fake_engine(engine)

    payload = cap.capture_inspection()

    # Two distinct layers, resolved independently.
    assert engine.neuron_calls[0]["layer"] == 16
    assert engine.head_calls[0] == {"layer": 18, "head": 12}
    assert payload["top_neurons"]["layer"] == 16
    assert payload["head_detail"]["layer"] == 18
    assert payload["head_detail"]["label"] == "L18H12"


def test_the_neuron_and_head_layers_are_not_confused(fake_engine):
    """The regression this pins.

    `list_neurons(8, ...)` and `head_detail(9, 9)` name different layers. A
    single resolved index reused for both produced a head inspection on L8H9 --
    a different measurement, with every other number in the artifact unchanged
    and therefore invisible.
    """
    engine = FakeEngine()
    fake_engine(engine)

    payload = cap.capture_inspection()

    assert engine.neuron_calls[0]["layer"] == 8
    assert engine.head_calls[0] == {"layer": 9, "head": 9}
    assert payload["head_detail"]["label"] == "L9H9"
    assert payload["head_detail"]["layer"] != payload["top_neurons"]["layer"]


def test_result_keys_carry_no_layer_digits(fake_engine):
    """A key naming a layer the run never visited outlives the mistake.

    `top_neurons_L8_by_in_norm` was the exact shape: the index lived in the key,
    so re-parameterising the sweep would have left the label claiming a layer
    that was not measured.
    """
    fake_engine(FakeEngine())
    payload = cap.capture_inspection()

    for key in payload:
        assert not any(ch.isdigit() for ch in key), (
            f"result key {key!r} embeds an index; indices belong in the payload")


def test_the_selection_is_recorded_in_the_artifact(fake_engine):
    fake_engine(FakeEngine())
    protocol = cap.declared_protocol()

    selection = protocol["selection"]
    assert selection["steering_layers"] == [6, 8, 10]
    assert selection["inspection_neuron_layer"] == 8
    assert selection["inspection_head_layer"] == 9
    assert selection["inspection_head"] == 9
    assert "Resolved from the loaded architecture" in selection["note"]


def test_declared_inputs_are_hashed_so_an_edit_shows_up(fake_engine):
    fake_engine(FakeEngine())
    first = cap.declared_protocol()["inputs_sha256"]

    assert first.startswith("sha256:")
    original = cap.IOI_TEMPLATES
    try:
        cap.IOI_TEMPLATES = [("a different template", " X", " Y")]
        changed = cap.declared_protocol()["inputs_sha256"]
    finally:
        cap.IOI_TEMPLATES = original

    assert changed != first, (
        "changing a declared prompt did not change the digest, so the artifact "
        "cannot show that the inputs moved")


def test_the_declared_inputs_are_recorded_not_just_hashed(fake_engine):
    fake_engine(FakeEngine())
    inputs = cap.declared_protocol()["inputs"]

    assert inputs["ioi_templates"][0][0].startswith("Then John and Mary")
    assert inputs["steer_positive"] == cap.STEER_POS
    assert inputs["steer_alphas"] == list(cap.STEER_ALPHAS)
    assert [c[0] for c in inputs["sanity_checks"]] == [
        n for n, *_r in cap.SANITY_CHECKS]


# ── Sweep completeness ──────────────────────────────────────────────────

def test_the_sweep_covers_every_head_and_records_the_expectation(fake_engine):
    fake_engine(FakeEngine())
    payload = cap.capture_head_sweep()

    assert payload["n_layers"] == 12
    assert payload["n_heads"] == 12
    assert payload["n_heads_expected"] == 144
    assert payload["n_heads_swept"] == len(payload["all_heads"]) == 144


def test_a_refused_head_fails_the_sweep_rather_than_shrinking_it(fake_engine):
    """`patch_head` returns a non-ok status rather than raising, and the loop
    used to `continue` past those. The sweep then reported a count that looked
    complete and ranked whatever survived.
    """
    fake_engine(FakeEngine(refuse_heads=[(3, 7)]))

    with pytest.raises(RuntimeError) as excinfo:
        cap.capture_head_sweep()

    message = str(excinfo.value)
    assert "143 of 144" in message
    assert "cannot be ranked as a complete one" in message


def test_a_sweep_over_a_different_model_covers_that_model(fake_engine):
    fake_engine(FakeEngine(dims={**GPT2_SMALL, "n_layers": 24, "n_heads": 16}))
    payload = cap.capture_head_sweep()

    assert payload["n_heads_expected"] == 24 * 16
    assert payload["n_heads_swept"] == 24 * 16


def test_model_dims_refuses_to_guess_when_the_engine_will_not_answer(monkeypatch):
    monkeypatch.setattr(cap, "engine", SimpleNamespace(
        info=lambda: {"status": "error", "error": "no model loaded"}))

    with pytest.raises(RuntimeError) as excinfo:
        cap.model_dims()

    assert "n_layers" in str(excinfo.value)
    assert "no model loaded" in str(excinfo.value)


# ── Reproduction manifest ───────────────────────────────────────────────

def test_the_environment_that_changes_the_numbers_is_recorded(fake_engine):
    fake_engine(FakeEngine())
    env = cap.capture_environment()

    for key in ("versions", "platform", "git_commit", "git_dirty",
                "script_sha256", "sampling"):
        assert key in env, f"{key} is missing from the reproduction manifest"
    assert set(env["versions"]) == {"python", "torch", "numpy", "transformers"}
    assert env["script_sha256"].startswith("sha256:")
    assert "deterministic forward pass" in env["sampling"]


def test_the_script_hash_covers_this_scripts_actual_bytes():
    env = cap.capture_environment()
    import hashlib
    expected = "sha256:" + hashlib.sha256(SCRIPT.read_bytes()).hexdigest()

    assert env["script_sha256"] == expected, (
        "the recorded script hash does not match the script that ran, so the "
        "artifact cites a revision of itself that does not exist")


def test_identity_carries_the_hashes_that_pin_the_checkpoint(fake_engine):
    fake_engine(FakeEngine())
    identity = cap.capture_identity()

    assert identity["weights_sha256"].startswith("sha256:")
    assert identity["config_sha256"].startswith("sha256:")
    assert identity["tokenizer_sha256"].startswith("sha256:")
    assert identity["revision"]
    assert identity["parameter_count"] == 124439808


# ── Structural ──────────────────────────────────────────────────────────

def test_no_capture_reaches_into_the_engines_private_accessors():
    """`engine._n_layers()` was how the sweep got its depth.

    The private accessor is not itself the problem -- it is that a private
    accessor can change or disappear without the script noticing, and the sweep
    would silently cover a different number of heads. Dimensions come from
    `model_dims()`, which goes through the public `info()`.
    """
    source = SCRIPT.read_text(encoding="utf-8")
    for private in ("engine._n_layers", "engine._n_heads", "engine._model",
                    "engine._tokenizer"):
        assert private not in source, (
            f"{private} is called; dimensions and identity should come from "
            f"model_dims() and model_identity()")


def test_the_sweep_label_does_not_assert_a_head_count():
    """`"144-head zero-ablation sweep"` described one checkpoint."""
    labels = {section: label for section, label, _name in cap.CAPTURES}
    assert labels["head_sweep"] == "full head zero-ablation sweep"
