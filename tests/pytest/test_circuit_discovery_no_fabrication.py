"""`circuit_discovery` must not invent a circuit, and provenance must be readable.

Two defects, one of them in code added earlier in this same effort.

1. The no-adapter fallback fabricated a circuit.
   `CircuitDiscoveryEngine.discover_circuit` with no connected adapter returned a
   complete synthetic result: a `Neuron` node labelled `L8_N402 (IOI)`, edges
   weighted 0.88 and 0.95 with confidences 0.94 and 0.98, `confidence: 0.95` and
   `runtime_ms: 150.0`. `graph.score` was 0.945 -- exactly the mean of the two
   invented confidences, so even the summary statistic was derived from the
   fabrication. `provenance` was `{}`, so nothing marked it.

   The shape was the problem rather than any single number: a caller reading
   `result["confidence"]` got 0.95, and a caller rendering `result["graph"]`
   drew a plausible IOI circuit containing a named neuron. `L8_N402` is the
   sharpest edge of it -- a specific claim about a specific neuron, invented.

2. The provenance label was unreadable by the evidence policy.
   `evidence_policy.provenance_of` resolves a report's `provenance` dict by
   checking `source`, `kind`, `type` and `status` in that order, and returning
   "unavailable" when it finds none. The convention in `provenance_block` nested
   the label under `provenance` instead, so a live run carrying
   `{"provenance": "live", ...}` reported `provenance_of(...) == "unavailable"`
   to the very policy meant to gate it.

   This was invisible for the unimplemented case, where "reference" is not "live"
   either way -- the broken shape produced no wrong verdict, only an unusable live
   one. That asymmetry is why it survived, and it was introduced by the ACDC and
   causal-scrubbing fixes earlier in this effort.
"""

from __future__ import annotations

import re

import pytest

from backend.agents.evidence_policy import provenance_of


from _weight_guard import skip_reason, weights_available


needs_weights = pytest.mark.skipif(
    not weights_available(), reason=skip_reason()
)


# ── The fabricated fallback ─────────────────────────────────────────────────

def test_no_adapter_returns_unavailable_not_a_synthetic_circuit():
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )

    result = CircuitDiscoveryEngine().discover_circuit()

    assert result["status"] == "unavailable"
    assert result["confidence"] is None
    assert result["runtime_ms"] is None
    assert result["measured"] is False
    assert result["validation_eligible"] is False
    assert result["publication_eligible"] is False
    assert result["graph"] == {"nodes": [], "edges": [], "score": None}
    assert result["statistics"] == {} and result["evidence"] == {}
    assert "No model adapter is connected" in result["reason"]


def test_no_specific_neuron_is_named_anywhere():
    """`L8_N402` was a specific invented claim about a specific neuron."""
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )

    blob = str(CircuitDiscoveryEngine().discover_circuit())
    assert "L8_N402" not in blob
    assert "N402" not in blob


def test_no_fabricated_weights_or_confidences_survive():
    """Check the data fields, not the whole record.

    The `reason` text legitimately names the values that were removed -- "it
    previously returned ... confidence 0.95" -- so scanning the serialised record
    flags the documentation of the fix as if it were the defect. An earlier
    version of this test did exactly that and failed on its own explanatory
    string.
    """
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )

    result = CircuitDiscoveryEngine().discover_circuit()

    payload = {
        "confidence": result["confidence"],
        "runtime_ms": result["runtime_ms"],
        "graph": result["graph"],
        "statistics": result["statistics"],
        "evidence": result["evidence"],
        "model_id": result["model_id"],
    }
    blob = str(payload)
    for invented in ("0.88", "0.94", "0.95", "0.98", "0.945"):
        assert invented not in blob, f"fabricated value {invented} still emitted"


def test_provenance_is_readable_by_the_evidence_policy():
    """A plain string, so `provenance_of` resolves it without a dict lookup."""
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )

    result = CircuitDiscoveryEngine().discover_circuit()
    assert result["provenance"] == "unavailable"
    assert provenance_of(result) == "unavailable"


# ── Error paths are labelled, not swallowed ─────────────────────────────────

def test_missing_dataset_is_reported_as_unavailable():
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    result = CircuitDiscoveryEngine(
        GPT2Adapter(variant="small", mock_mode=True)
    ).discover_circuit(dataset_name="no_such_dataset")

    assert result["status"] == "unavailable"
    assert "no_such_dataset" in result["reason"]


def test_unknown_algorithm_is_reported_as_unavailable():
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    result = CircuitDiscoveryEngine(
        GPT2Adapter(variant="small", mock_mode=True)
    ).discover_circuit(algorithm_name="not_a_registered_algorithm")

    assert result["status"] == "unavailable"
    assert "not_a_registered_algorithm" in result["reason"]
    assert "confidence" in result and result["confidence"] is None


def test_a_raising_algorithm_is_unavailable_not_a_bare_error_dict():
    """Previously `except Exception: return {"error": str(e)}`.

    That dict has no `status`, no `confidence` and no `provenance`, so a consumer
    branching on `confidence` could not tell a failure from a result with no
    score. It is now the same shape as every other unavailable record.
    """
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    engine = CircuitDiscoveryEngine(GPT2Adapter(variant="small", mock_mode=True))

    class Boom:
        def run(self, dataset=None, config=None):
            raise RuntimeError("synthetic failure for the test")

    import backend.interpretability.discovery.circuit_discovery as mod
    original = mod.get_algorithm
    mod.get_algorithm = lambda name, adapter: Boom()
    try:
        result = engine.discover_circuit()
    finally:
        mod.get_algorithm = original

    assert result["status"] == "unavailable"
    assert "synthetic failure for the test" in result["reason"]
    assert result["error"].startswith("RuntimeError")
    assert result["measured"] is False


# ── Execution and evidence are separate ─────────────────────────────────────

def test_mock_weights_report_completed_but_unmeasured():
    """The algorithm ran and returned a report; nothing was measured.

    `status` says the algorithm ran. `measured` says whether it took
    measurements. Setting `status` from `measured` made a run that evaluated 144
    heads report "unavailable" purely because the whole-circuit fidelity needs
    `io_id`/`subject_id`, which the bundled dataset does not carry.
    """
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    result = CircuitDiscoveryEngine(
        GPT2Adapter(variant="small", mock_mode=True)
    ).discover_circuit()

    assert result["status"] == "completed"
    assert result["measured"] is False
    assert result["statistics"]["total_evaluations"] == 0


@needs_weights
def test_live_run_is_marked_live_in_a_form_the_policy_can_read():
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    result = CircuitDiscoveryEngine(
        GPT2Adapter(variant="small", mock_mode=False)
    ).discover_circuit()

    assert result["status"] == "completed"
    assert result["measured"] is True
    assert result["statistics"]["total_evaluations"] > 0

    # The regression: this returned "unavailable" while the report carried
    # `"provenance": "live"` in the same dict.
    assert provenance_of(result) == "live", (
        "the evidence policy cannot read this report's provenance"
    )
    assert result["provenance"]["source"] == "live"


@needs_weights
def test_live_edge_confidences_are_measured_not_constant():
    """The fabricated fallback used one confidence on every edge."""
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    result = CircuitDiscoveryEngine(
        GPT2Adapter(variant="small", mock_mode=False)
    ).discover_circuit()

    confidences = {
        e["confidence"] for e in result["graph"]["edges"]
        if e.get("confidence") is not None
    }
    assert len(confidences) > 1, (
        f"every edge carries the same confidence {confidences}"
    )


@needs_weights
def test_eligibility_is_not_asserted_without_an_explicit_opt_in():
    """A measured sweep is not a scientifically adequate one.

    Setting eligibility here would compress "we measured something" into "this may
    support a publication", which is the distinction the evidence policy exists
    to keep open.
    """
    from backend.interpretability.discovery.circuit_discovery import (
        CircuitDiscoveryEngine,
    )
    from backend.science.models.gpt2_adapter import GPT2Adapter

    result = CircuitDiscoveryEngine(
        GPT2Adapter(variant="small", mock_mode=False)
    ).discover_circuit()

    assert result["measured"] is True
    assert result["validation_eligible"] is False
    assert result["publication_eligible"] is False


# ── Cross-cutting: provenance shape ─────────────────────────────────────────

@pytest.mark.parametrize(
    "label,payload,expected",
    [
        ("plain string", {"provenance": "live"}, "live"),
        ("source key", {"provenance": {"source": "live"}}, "live"),
        ("kind key", {"provenance": {"kind": "live"}}, "live"),
        ("type key", {"provenance": {"type": "live"}}, "live"),
        ("status key", {"provenance": {"status": "live"}}, "live"),
        # The shape that caused the regression: a dict with no key
        # `provenance_of` recognises.
        ("unrecognised dict", {"provenance": {"provenance": "live"}}, "unavailable"),
    ],
)
def test_provenance_of_recognises_exactly_the_documented_shapes(label, payload, expected):
    """Pins which provenance shapes the evidence policy can and cannot read.

    The failing row is not a bug in `provenance_of` -- it is a guard against
    writing a dict whose label the policy cannot see, which is how a live
    measurement becomes invisible to its own gate.
    """
    assert provenance_of(payload) == expected, label


def test_unimplemented_algorithm_provenance_is_readable_as_reference():
    """The shared `provenance_block` must carry a key the policy recognises."""
    from backend.interpretability.discovery.algorithms.base_algorithm import (
        DiscoveryAlgorithm,
    )

    class Unimplemented(DiscoveryAlgorithm):
        implements_published_method = False
        not_implemented_reason = "test double"

        def run(self, dataset, config=None):
            raise NotImplementedError

    block = Unimplemented(None).provenance_block()
    assert block["source"] == "reference"
    assert block["provenance"] == "reference", (
        "the nested key is kept for human readers"
    )


# ── Source guard ────────────────────────────────────────────────────────────

def test_circuit_discovery_source_has_no_fabricated_values():
    from source_assert import executable_source

    import backend.interpretability.discovery.circuit_discovery as cd

    src = executable_source(cd)

    assert "L8_N402" not in src, "the invented neuron is back in the source"
    assert "Fallback Mock for UI" not in src
    assert not re.search(r'"confidence":\s*0\.\d', src)
    assert not re.search(r'"weight":\s*0\.\d', src)
    assert not re.search(r'"runtime_ms":\s*\d+\.\d', src), (
        "a literal runtime is being reported as a measurement"
    )
