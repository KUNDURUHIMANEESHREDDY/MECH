"""Fabricated results must not survive in engines that measure nothing.

Seams (all previously returned hardcoded numbers as if measured):

  * ``backend.interpretability.semantics.mechanistic_report`` -- a fixed
    four-step causal narrative with ``causal_confidence`` 0.95.
  * ``backend.core.provenance_viewer`` -- a fixed lineage asserting
    ``verified_lineage: True``.
  * ``backend.validation.peer_review`` -- a fixed ``decision: "Accept"`` and
    ``score: 9.2``.
  * ``backend.validation.confidence_engine`` -- assumed defaults plus a
    hardcoded ``statistical_rigor: "High"``.
  * ``backend.interpretability.inspectors.prediction_inspector`` /
    ``feature_inspector`` -- invented distributions and feature records.
  * ``backend.interpretability.repository.feature_repository`` -- two seeded
    feature records.
  * ``backend.core.evidence_graph`` -- seven seeded evidence nodes reported in
    ``AIScientistEngine`` campaign summaries.
  * ``backend.runtime.adaptive.runtime_simulator`` -- literals 94.0 and True.
  * ``backend.runtime.adaptive.execution_episodes`` -- a seeded successful run.
  * ``backend.runtime.orchestration.resource_optimizer`` -- an unlabelled
    policy table.
  * ``backend.ui.representation_atlas`` -- invented neuron/citation claims and
    a "VERIFIED" status.

Each test asserts the *observation* (the real input is read, the refusal names
itself) rather than only the outcome string, and carries a control so it cannot
pass by the module being broken or the call being malformed. See the sibling
pattern in ``test_process_denylist_policy.py``, where a TypeError satisfied a
reachability assertion for the wrong reason.
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.core import evidence_graph as eg  # noqa: E402
from backend.core.provenance_viewer import ProvenanceViewerEngine  # noqa: E402
from backend.interpretability.repository.feature_repository import (  # noqa: E402
    FeatureRepository,
    get_feature_repository,
)
from backend.interpretability.semantics.mechanistic_report import (  # noqa: E402
    MechanisticReportEngine,
)
from backend.runtime.adaptive.execution_episodes import (  # noqa: E402
    ExecutionEpisodesEngine,
)
from backend.runtime.adaptive.runtime_simulator import (  # noqa: E402
    RuntimeSimulatorEngine,
)
from backend.runtime.orchestration.resource_optimizer import (  # noqa: E402
    ResourceOptimizerEngine,
)
from backend.ui.representation_atlas import RepresentationAtlas  # noqa: E402
from backend.validation.confidence_engine import (  # noqa: E402
    ScientificConfidenceEngine,
)
from backend.validation.peer_review import AutomatedPeerReviewer  # noqa: E402
from interpretability.inspectors.feature_inspector import (  # noqa: E402
    FeatureInspector,
)
from interpretability.inspectors.prediction_inspector import (  # noqa: E402
    PredictionInspector,
)
from backend.interpretability.discovery.algorithms.feature_universality import (  # noqa: E402
    FeatureUniversalityAlgorithm,
)
from backend.interpretability.discovery.discovery_result import (  # noqa: E402
    DiscoveryResultDTO,
)
from backend.research_platform.autonomous.research_graph import (  # noqa: E402
    TypedResearchGraph,
)
from backend.research_platform.autonomous.research_memory import (  # noqa: E402
    ResearchMemoryEngine,
)
from backend.research_platform.meta.campaign_embeddings import (  # noqa: E402
    CampaignEmbeddingsEngine,
)
from backend.science.models import model_manager as _model_manager  # noqa: E402
from backend.science.models.gpt2_adapter import GPT2Adapter  # noqa: E402

# Values that were previously returned as findings. Whole-value literals are
# checked for equality and the rest as substrings: `"VERIFIED"` is a substring
# of the replacement `"UNVERIFIED"`, so a substring check for it would fail on
# the correct value.
FABRICATED_EXACT_VALUES = (
    "Accept",     # peer review verdict
    "VERIFIED",   # atlas verification status
    "High",       # statistical_rigor rating
)
FABRICATED_SUBSTRINGS = (
    "9.2",             # peer review score
    "0.95",            # causal_confidence / assumed reproducibility
    "94.0",            # runtime simulator confidence_pct
    "1420.5",          # seeded episode throughput
    "4.12",            # seeded feature / connected neuron activation
    "L8N402",          # invented neuron id (atlas)
    "L8_N402",         # invented neuron id (evidence graph / report)
    "Bricken",         # invented citation
    "IOI Prompts v1",  # invented dataset
    "ArXiv Package",   # invented publication
    "ep_ioi_001",      # seeded episode id
)

_EXPLANATORY_KEYS = {"reason", "status"}


def _result_fields(payload):
    """The fields a consumer reads as results, minus the refusal narration."""
    assert isinstance(payload, dict), f"expected a dict, got {type(payload)}"
    return {k: v for k, v in payload.items() if k not in _EXPLANATORY_KEYS}


def _iter_scalars(value):
    """Every leaf value in a nested payload."""
    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_scalars(nested)
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _iter_scalars(nested)
    else:
        yield value


def _assert_no_fabricated_literals(payload, where):
    scalars = list(_iter_scalars(_result_fields(payload)))
    for scalar in scalars:
        assert scalar not in FABRICATED_EXACT_VALUES, (
            f"{where} still returns the fabricated value {scalar!r} in a "
            f"result field"
        )
    for literal in FABRICATED_SUBSTRINGS:
        for scalar in scalars:
            assert literal not in str(scalar), (
                f"{where} still returns the fabricated literal {literal!r} in "
                f"a result field: {scalar!r}"
            )


def _assert_self_declaring(payload, where):
    assert payload["measured"] is False, f"{where} must not claim a measurement"
    assert payload["validation_eligible"] is False, where
    assert payload["publication_eligible"] is False, where
    assert payload["reason"], f"{where} must state why it returned nothing"


# ── mechanistic report ────────────────────────────────────────────────────

def test_mechanistic_report_does_not_invent_a_narrative():
    res = MechanisticReportEngine().generate_mechanistic_report(
        prompt="The capital of France is")

    assert res["explanation_text"] is None
    assert res["circuit_components"] == []
    assert res["causal_confidence"] is None
    _assert_self_declaring(res, "mechanistic report")
    _assert_no_fabricated_literals(res, "mechanistic report")

    # Control: the echoed inputs are read, so the empty explanation is a
    # refusal rather than an unread argument.
    assert res["prompt"] == "The capital of France is"
    assert res["target_token"] == " Paris"


# ── provenance viewer ─────────────────────────────────────────────────────

def test_provenance_viewer_never_asserts_verified_lineage():
    res = ProvenanceViewerEngine().get_lineage("disc_anything")

    assert res["lineage"] == []
    assert res["verified_lineage"] is False
    _assert_self_declaring(res, "provenance viewer")
    _assert_no_fabricated_literals(res, "provenance viewer")

    # Control: the id is echoed, so "no lineage" is a lookup result and not an
    # ignored argument.
    assert res["discovery_id"] == "disc_anything"
    assert ProvenanceViewerEngine().get_lineage("d2")["discovery_id"] == "d2"


# ── peer review ───────────────────────────────────────────────────────────

def test_peer_review_does_not_return_an_accept():
    res = AutomatedPeerReviewer().review_manuscript("Any title at all")

    assert res["decision"] is None
    assert res["score"] is None
    assert res["strengths"] == []
    assert res["suggestions"] == []

    # An automated review must never occupy the human-review slot that the
    # validation gates treat as authoritative.
    assert res["is_human_peer_review"] is False

    _assert_self_declaring(res, "peer review")
    _assert_no_fabricated_literals(res, "peer review")

    # Control: the title is echoed, so the missing verdict is not a signature
    # mismatch or an unread argument.
    assert res["title"] == "Any title at all"

    # A caller testing the verdict gets a falsy value, which is the correct
    # answer for a review that did not happen.
    assert not res["decision"]


# ── confidence engine ─────────────────────────────────────────────────────

def test_confidence_engine_refuses_when_inputs_are_missing():
    res = ScientificConfidenceEngine().compute_confidence()

    assert res["confidence_score"] is None
    assert res["statistical_rigor"] is None
    assert res["uncertainty_interval"] is None
    assert res["inputs_assumed"] is True
    assert set(res["assumed_inputs"]) == {"reproducibility_score", "p_value"}
    _assert_self_declaring(res, "confidence engine (no inputs)")
    _assert_no_fabricated_literals(res, "confidence engine (no inputs)")


def test_confidence_engine_names_only_the_missing_input():
    """A half-supplied call must not silently default the other term."""
    res = ScientificConfidenceEngine().compute_confidence(
        reproducibility_score=0.9)

    assert res["assumed_inputs"] == ["p_value"]
    assert res["confidence_score"] is None

    # Control: supplying both yields a score, so the refusal above is about the
    # missing input and not about the method being broken.
    full = ScientificConfidenceEngine().compute_confidence(
        reproducibility_score=0.9, p_value=0.5)
    assert full["confidence_score"] is not None
    assert full["inputs_assumed"] is False
    # ...and rigor is still not rated: two metrics cannot establish it.
    assert full["statistical_rigor"] is None
    _assert_self_declaring(full, "confidence engine (computed)")


# ── inspectors ────────────────────────────────────────────────────────────

def test_feature_inspector_does_not_invent_a_feature_record():
    res = FeatureInspector().inspect(feature_id=1402)

    assert res["activation"] is None
    assert res["label"] is None
    assert res["connected_neurons"] == []
    assert res["dataset_examples"] == []
    assert res["statistics"] == {}
    _assert_self_declaring(res, "feature inspector")
    _assert_no_fabricated_literals(res, "feature inspector")

    # Control: the requested id is honoured, so the empty record is a refusal
    # and not an unread argument.
    assert res["feature_id"] == 1402
    assert FeatureInspector().inspect(feature_id=7)["feature_id"] == 7


def test_prediction_inspector_payload_carries_no_fabrication():
    """Guard the same property from the predictive side."""
    res = PredictionInspector().inspect(prompt="p", top_k=5)

    assert res["predictions"] == []
    assert res["entropy"] is None
    assert res["logit_lens"] is None
    _assert_self_declaring(res, "prediction inspector")
    _assert_no_fabricated_literals(res, "prediction inspector")


# ── feature repository ────────────────────────────────────────────────────

def test_feature_repository_is_empty_by_default():
    repo = FeatureRepository()

    assert repo.search_features() == []
    assert repo.search_features(query="indirect object") == []
    # The previously seeded ids must not resolve to a record.
    assert repo.get_feature(1402) is None
    assert repo.get_feature(789) is None

    # Control: the read paths work and the singleton is constructed, so an
    # empty result is an empty store and not a broken accessor.
    assert isinstance(get_feature_repository(), FeatureRepository)
    assert repo._features == {}


# ── evidence graph ────────────────────────────────────────────────────────

def test_evidence_graph_does_not_seed_demo_evidence():
    graph = eg.TraceableEvidenceGraph()

    assert graph.nodes == []
    assert graph.edges == []
    # AIScientistEngine reports this count in its campaign summary, where a
    # non-zero value would advertise evidence that was never measured.
    assert graph.to_dict()["nodes_count"] == 0
    assert graph.to_dict()["edges_count"] == 0


def test_demo_graph_is_opt_in_and_labelled():
    demo = eg.TraceableEvidenceGraph.demo()

    assert len(demo.nodes) == 7
    assert len(demo.edges) == 6
    for node in demo.nodes:
        assert node["measured"] is False, node
        assert node["provenance"] == "reference", node

    # Control: the fixture is opt-in only, so constructing the class normally
    # must not produce it.
    assert eg.TraceableEvidenceGraph().nodes == []


def test_from_run_still_builds_a_real_graph():
    """Control: the class remains functional, so the empty default is a
    deliberate refusal rather than a broken constructor."""
    graph = eg.TraceableEvidenceGraph.from_run(
        "rab12cd34ef56", "goal",
        [{"node": "discover", "agent": "a", "status": "ok", "result": {}}],
    )
    assert len(graph.nodes) >= 2
    assert graph.to_dict()["nodes_count"] >= 2


# ── runtime simulator ─────────────────────────────────────────────────────

def test_runtime_simulator_does_not_report_a_hardcoded_confidence():
    res = RuntimeSimulatorEngine().simulate_execution()

    assert res["confidence_pct"] is None
    assert res["simulation_is_estimate"] is True
    _assert_self_declaring(res, "runtime simulator")
    _assert_no_fabricated_literals(res, "runtime simulator")


def test_simulation_passed_is_derived_not_unconditional():
    """The verdict must be able to come out False.

    ``simulation_passed`` was the literal ``True``, so no input could make the
    field report a failure. FP32 doubles the base VRAM to 16.8 GB, which is
    the same comparison that selects the safety margin.
    """
    engine = RuntimeSimulatorEngine()

    fits = engine.simulate_execution(precision="FP16")
    assert fits["simulation_passed"] is True
    assert fits["recommended_safety_margin"] == "Sufficient VRAM"

    over = engine.simulate_execution(precision="FP32")
    assert over["simulation_passed"] is False, (
        "simulation_passed cannot be derived if no input makes it False")
    assert over["recommended_safety_margin"] == "Consider INT8 Quantization"

    # The verdict and the advice must agree on both paths.
    assert (fits["simulated_vram_gb"] < 16.0) is fits["simulation_passed"]
    assert (over["simulated_vram_gb"] < 16.0) is over["simulation_passed"]


# ── execution episodes ────────────────────────────────────────────────────

def test_execution_episodes_start_empty():
    engine = ExecutionEpisodesEngine()

    assert engine.episodes == {}
    # The seeded episode must not be reachable under its old id.
    assert "ep_ioi_001" not in engine.episodes

    # Control: the recorder exists, so the store is empty because nothing was
    # recorded rather than because recording is unimplemented.
    assert callable(engine.record_episode)


# ── resource optimizer ────────────────────────────────────────────────────

def test_resource_optimizer_labels_its_policy_source():
    res = ResourceOptimizerEngine().optimize_resources(strategy="Speed")

    assert res["source"] == "static_policy_table"
    assert res["measured"] is False
    assert res["validation_eligible"] is False
    assert res["reason"]

    # Control: the policy row itself is preserved, so labelling did not erase
    # the behaviour callers depend on.
    assert res["gpu_target"] == "NVIDIA H100"
    assert res["precision"] == "FP16"


def test_resource_optimizer_reports_the_row_it_actually_used():
    engine = ResourceOptimizerEngine()

    assert engine.optimize_resources("Memory")["strategy"] == "Memory"
    # An unknown strategy falls back, and the fallback is now visible instead
    # of being implied by the absence of the caller's name.
    fallback = engine.optimize_resources("Nonsense")
    assert fallback["strategy"] == "Balanced"
    assert fallback["gpu_target"] == "NVIDIA A100"


# ── representation atlas ──────────────────────────────────────────────────

def test_representation_atlas_does_not_claim_verification():
    res = RepresentationAtlas.get_node_details(
        ["Knowledge", "Geography", "Cities", "Paris"])

    assert res["verification_status"] == "UNVERIFIED"
    assert res["associated_experiments"] == []
    assert res["measured"] is False
    assert res["reason"]

    # The invented neuron/citation pairing must be gone from the payload.
    assert res["data"].get("neuron") is None
    assert res["data"].get("paper") is None
    assert res["data"].get("association_measured") is False

    # Control: navigation still works, so the taxonomy is intact and only the
    # unmeasured claims were removed.
    assert res["node"] == "Paris"
    countries = RepresentationAtlas.get_node_details(
        ["Knowledge", "Geography", "Countries"])
    assert countries["data"]["concept_id"] == "concept_001"
    # ...and an unknown path still reports not-found rather than inventing one.
    assert RepresentationAtlas.get_node_details(["Nope"])["error"] == "Node not found"


# ── cross-cutting sweep ───────────────────────────────────────────────────

def test_no_fixed_fabricated_literal_survives_in_any_result_field():
    """One sweep over every engine above, so a reintroduced literal fails here
    even if the field-specific assertion above is edited away."""
    payloads = {
        "mechanistic_report":
            MechanisticReportEngine().generate_mechanistic_report("p"),
        "provenance_viewer": ProvenanceViewerEngine().get_lineage("d"),
        "peer_review": AutomatedPeerReviewer().review_manuscript("t"),
        "confidence_engine": ScientificConfidenceEngine().compute_confidence(),
        "feature_inspector": FeatureInspector().inspect(feature_id=1),
        "prediction_inspector": PredictionInspector().inspect(prompt="p"),
        "runtime_simulator": RuntimeSimulatorEngine().simulate_execution(),
        "resource_optimizer": ResourceOptimizerEngine().optimize_resources(),
        "representation_atlas": RepresentationAtlas.get_node_details(
            ["Knowledge", "Geography", "Cities", "Paris"]),
    }
    for where, payload in payloads.items():
        _assert_no_fabricated_literals(payload, where)


# ── feature universality algorithm ────────────────────────────────────────

def test_feature_universality_refuses_to_report_a_discovery():
    """The registered algorithm must not invent cross-model matches.

    It returned eight fixed concepts with fixed similarity scores ("French
    Cities" 0.94 upward), generated feature ids as ``100 + i * 15``, and set
    ``confidence`` to their average -- as a ``DiscoveryReport`` with a
    populated graph. ``distributed_campaign``, ``dag_discovery_planner``,
    ``discovery_planner`` and ``autonomous_paper_replicator`` schedule this
    algorithm by name, so the fabrication reached campaigns.
    """
    report = FeatureUniversalityAlgorithm(adapter=None).run({})
    payload = report.to_dict()

    # No score, and no graph to read one off.
    assert payload["confidence"] is None
    assert payload["graph"] == {"nodes": [], "edges": [], "score": None}
    assert payload["evidence"]["universal_alignments"] == []
    assert payload["statistics"]["matched_pairs_count"] == 0
    assert payload["statistics"]["avg_universality_score"] is None
    assert payload["statistics"]["measured"] is False

    # The provenance block must be readable by the evidence gate, which looks
    # for `source` first. A block that only nests the label is invisible to it.
    assert payload["provenance"]["source"] == "reference"
    assert payload["provenance"]["measured"] is False
    assert payload["provenance"]["validation_eligible"] is False
    assert payload["provenance"]["publication_eligible"] is False
    assert payload["provenance"]["reason"]

    _assert_no_fabricated_literals(payload["statistics"], "feature universality")
    _assert_no_fabricated_literals(payload["graph"], "feature universality")

    # Control: the algorithm still declares itself, and the report still
    # identifies which models a real run would have compared, so the refusal is
    # informative rather than an empty object.
    assert FeatureUniversalityAlgorithm.implements_published_method is False
    assert FeatureUniversalityAlgorithm.not_implemented_reason
    assert payload["algorithm"] == "feature_universality"
    assert len(payload["statistics"]["aligned_models"]) == 2


def test_feature_universality_no_longer_substitutes_a_default_prompt():
    """A dataset that supplies nothing must not become a scorable prompt.

    The run used to fall back to ``"Paris is the capital of France"`` so an
    empty dataset still produced a report with a score.
    """
    report = FeatureUniversalityAlgorithm(adapter=None).run({})
    assert "Paris is the capital of France" not in repr(report.to_dict())


# ── discovery DTO provenance ──────────────────────────────────────────────

def test_discovery_dto_does_not_invent_provenance():
    """An omitted provenance must not become a specific prompt and model.

    ``DiscoveryResultDTO`` defaulted ``provenance`` to
    ``{"prompt": "The capital of France is", "model": "GPT-2 Small"}``, which
    asserts that a particular prompt ran against a particular model.
    """
    dto = DiscoveryResultDTO(
        discovery_id="d1", discovery_type="circuit", title="t", evidence=[])

    assert dto.provenance["source"] == "unavailable"
    assert dto.provenance["measured"] is False
    assert dto.provenance["reason"]
    assert "prompt" not in dto.provenance
    assert "model" not in dto.provenance

    # Control: supplied provenance is preserved verbatim, so the default above
    # is a refusal and not the argument being ignored.
    supplied = {"source": "live", "model": "gpt2", "revision": "abc"}
    explicit = DiscoveryResultDTO(
        discovery_id="d2", discovery_type="circuit", title="t", evidence=[],
        provenance=supplied)
    assert explicit.provenance == supplied

    # An explicitly empty dict is a supplied (if useless) value, not absence.
    empty = DiscoveryResultDTO(
        discovery_id="d3", discovery_type="circuit", title="t", evidence=[],
        provenance={})
    assert empty.provenance == {}


# ── campaign embeddings ───────────────────────────────────────────────────

def test_campaign_embeddings_index_starts_empty():
    """No campaign is seeded, so a replay cannot find one that never ran.

    A `camp_s6_ioi` entry with a fixed vector and an invented IOI summary used
    to be installed in `__init__`. `MetaResearchEngine.run_closed_feedback_loop`
    defaults `campaign_id` to that same id, so the seeded record was the one a
    default replay reached.
    """
    engine = CampaignEmbeddingsEngine()
    assert engine.index == {}

    # An empty index must produce no results, not fabricated ones.
    assert engine.find_similar_campaigns("Indirect Object Identification") == []


def test_campaign_similarity_is_not_presented_as_semantic():
    """A hash-vector cosine must not be readable as a relevance measure."""
    engine = CampaignEmbeddingsEngine()
    engine.index_campaign("c1", "IOI circuit discovery", "Ran ACDC on GPT-2.")

    results = engine.find_similar_campaigns("IOI circuit discovery")
    assert len(results) == 1
    hit = results[0]
    assert hit["campaign_id"] == "c1"

    # The score survives so ordering stays reproducible, but every reader that
    # would treat it as evidence finds a flag saying it is not.
    assert isinstance(hit["similarity_score"], float)
    assert hit["similarity_is_semantic"] is False
    assert hit["measured"] is False
    assert hit["source"] == "mock_hash_vector"
    assert hit["validation_eligible"] is False
    assert hit["publication_eligible"] is False
    assert hit["reason"]

    # Control: the same engine still indexes and retrieves a real campaign, so
    # the assertions above describe a labelled result and not an empty one.
    engine.index_campaign("c2", "Induction heads", "Ran head detection.")
    assert len(engine.find_similar_campaigns("Induction heads", top_k=5)) == 2


# ── adapter simulation flag ───────────────────────────────────────────────

def test_adapter_declares_simulation_when_weights_cannot_load(monkeypatch):
    """An adapter that fabricates output must set `simulated`, not just `mock_mode`.

    `ModelAdapter._load_model` degrades gracefully when weights are
    unavailable: it flips `spec.mock_mode` and serves mock fixtures. It left
    `simulated` at False, and `simulated` is the flag whose own comment says
    downstream provenance checks read it as "not a measurement" -- so a load
    failure produced output that read as a real one.
    """
    def _no_weights(self, hf_repo_id, revision="main"):
        raise RuntimeError("weights unavailable for this test")

    monkeypatch.setattr(
        _model_manager.ModelManager, "get_model_and_tokenizer", _no_weights)

    adapter = GPT2Adapter(variant="small", mock_mode=False)

    # Control: the fallback actually happened, so the assertions below describe
    # a degraded adapter rather than a mock_mode=True one that never tried.
    assert adapter.spec.mock_mode is True
    assert adapter.model_fingerprint() is None

    assert adapter.simulated is True
    assert adapter.simulation_reason
    assert "weights unavailable for this test" in adapter.simulation_reason

    # A caller asking for the spec is told, rather than having to infer it.
    assert adapter.get_model_spec()["mock_mode"] is True


# ── typed research graph ──────────────────────────────────────────────────

def test_typed_research_graph_starts_empty():
    """Seeded entities must not be counted alongside real ones.

    The constructor seeded six entities and five edges describing one IOI
    result, including the evidence node "Logit delta -4.2 on ' Paris' upon
    zeroing L8_N402" and the conclusion that the neuron is essential. Nothing
    marked them as an illustration, and `AutonomousResearchEngine` appends its
    real nodes to the same graph.
    """
    graph = TypedResearchGraph()
    assert graph.nodes == []
    assert graph.edges == []

    payload = graph.to_dict()
    assert payload["nodes_count"] == 0
    assert payload["edges_count"] == 0

    # Control: a node added by a caller is counted, so the zero above is an
    # empty graph and not a broken counter.
    graph.add_node("g_0", "Question", "Why does GPT-2 predict Paris?")
    assert graph.to_dict()["nodes_count"] == 1


def test_typed_research_graph_demo_is_opt_in_and_labelled():
    """The illustrative graph is requested explicitly and says what it is."""
    graph = TypedResearchGraph.demo()
    assert len(graph.nodes) == 6
    assert len(graph.edges) == 5

    for node in graph.nodes:
        assert node["measured"] is False
        assert node["provenance"] == "reference"

    # Control: the same content is still available, so the default-empty
    # assertion below is about opting in, not about deleting the fixture.
    assert any(node["id"] == "ev_1" for node in graph.nodes)
    assert TypedResearchGraph().nodes == []


# ── research memory ───────────────────────────────────────────────────────

def test_research_memory_starts_empty_and_does_not_default_a_score():
    """A memory supplied without a score must carry no score.

    Two memories were seeded, one asserting a measured logit shift with
    `utility_score` 0.95. `record_memory` also defaulted the score to 0.5, so a
    caller that supplied none still got a number -- `AutonomousResearchEngine`
    is such a caller, and its comment records that the 0.92 it used to pass was
    invented.
    """
    engine = ResearchMemoryEngine()
    assert engine.memories == []

    mem = engine.record_memory("autonomous_discovery", "Processed goal 'g'.")
    assert mem["utility_score"] is None
    assert mem["measured"] is False
    assert mem["reason"]
    assert len(engine.list_memories()) == 1

    # Control: a supplied score is kept and marked measured, so the assertions
    # above do not merely show the argument being ignored.
    scored = engine.record_memory("x", "y", utility_score=0.8)
    assert scored["utility_score"] == 0.8
    assert scored["measured"] is True
    assert scored["reason"] == ""
