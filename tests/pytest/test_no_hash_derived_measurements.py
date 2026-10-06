"""Three places where a plausible-looking number stood in for a measurement.

Each of these produced output shaped like a result, in a plausible range, with
no measurement behind it. None of them was a stub returning an obvious constant,
which is the failure mode this repository is built to catch and the one it
already tests for. All three looked like findings.

1. `RepresentationBenchmark.run_benchmark` ignored its argument and computed
   purity and completeness from `hash()` of its own task names, so it scored
   every task between 0.88 and 0.97 on every run and `overall_rigor` landed near
   0.85 against a 0.75 threshold -- passing by construction.

2. `ProvenanceService.record_provenance` wrote
   `f"sha256_{hash(...) & 0xffffffff:08x}"`: eight hex characters of salted
   SipHash, labelled as a SHA-256, in the one field whose job is to let someone
   check whether two records describe the same run.

3. `sae.loader._load_tensors` called `torch.load(..., weights_only=False)` on a
   checkpoint path, which is a pickle load and therefore arbitrary code
   execution on anything it can be handed.

The determinism test below is the one that matters most. The hash-derived
benchmark returned *stable* numbers within a single process, so a test that ran
it twice in one session would have passed. Python randomises string hashing per
process, so the numbers moved between runs while looking like measurements --
which defeats reproducibility checking as well as fabricating. That is why these
checks spawn processes with different hash seeds rather than calling twice.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _probe(source: str, *args: str, seed: str = "0") -> str:
    """Run `source` in a subprocess under a given PYTHONHASHSEED."""
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
    out = subprocess.run(
        [sys.executable, "-c", source, *args],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=180,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def _under_seeds(source: str, *args: str) -> list:
    return [_probe(source, *args, seed=seed)
            for seed in ("0", "1", "12345")]


# ── The representation benchmark ─────────────────────────────────────────────

def test_no_assignments_means_no_scores():
    from backend.interpretability.discovery.representation_benchmark import (
        RepresentationBenchmark,
    )

    result = RepresentationBenchmark().run_benchmark({})
    assert result["provenance"] == "unavailable"
    assert result["overall_rigor"] is None, (
        "a benchmark with nothing to score reported an overall figure")
    assert result["domain_scores"] == {}
    assert result["tasks_measured"] == 0
    assert result["publication_eligible"] is False


def test_empty_assignments_are_rejected_rather_than_scored_as_zero():
    from backend.interpretability.discovery.representation_benchmark import (
        RepresentationBenchmark,
    )

    benchmark = RepresentationBenchmark()
    task = next(t for t in benchmark.tasks if t.domain == "Geography")
    result = benchmark.run_benchmark({
        task.target_concept: {"concept": "capital", "assignments": []},
    })
    assert result["provenance"] == "unavailable"
    assert result["domain_scores"] == {}


def test_scores_are_computed_from_the_assignments():
    """A worked case, so the arithmetic is pinned rather than trusted.

    Three items in cluster "cl1", two of which really are capitals:

        purity       = 2 correct of 3 in the matched cluster = 0.6667
        completeness = 2 found of 2 that truly are capitals  = 1.0
    """
    from backend.interpretability.discovery.representation_benchmark import (
        RepresentationBenchmark,
    )

    benchmark = RepresentationBenchmark()
    task = next(t for t in benchmark.tasks if t.domain == "Geography")
    result = benchmark.run_benchmark({
        task.target_concept: {
            "concept": "capital",
            "assignments": [["cl1", "capital"], ["cl1", "capital"],
                            ["cl1", "river"]],
        },
    })

    score = result["domain_scores"][task.target_concept]
    assert score["purity"] == pytest.approx(2 / 3, abs=1e-4)
    assert score["completeness"] == pytest.approx(1.0)
    assert score["matched_cluster"] == "cl1"
    assert score["n_assigned"] == 3
    assert score["n_concept_items"] == 2


def test_a_clean_clustering_scores_one():
    from backend.interpretability.discovery.representation_benchmark import (
        RepresentationBenchmark,
    )

    benchmark = RepresentationBenchmark()
    task = next(t for t in benchmark.tasks if t.domain == "Geography")
    result = benchmark.run_benchmark({
        task.target_concept: {
            "concept": "capital",
            "assignments": [["cl1", "capital"], ["cl1", "capital"],
                            ["cl2", "river"], ["cl2", "river"]],
        },
    })
    assert result["domain_scores"][task.target_concept]["overall"] == 1.0


def test_a_contaminated_cluster_scores_low():
    """The direction the old implementation could never produce.

    Purity is 1 of 3 and completeness is 1 of 1, so overall is 0.667. The
    hash-derived version returned 0.74-0.88 for this task under every possible
    input, including an empty one.
    """
    from backend.interpretability.discovery.representation_benchmark import (
        RepresentationBenchmark,
    )

    benchmark = RepresentationBenchmark()
    task = next(t for t in benchmark.tasks if t.domain == "Geography")
    result = benchmark.run_benchmark({
        task.target_concept: {
            "concept": "capital",
            "assignments": [["cl1", "capital"], ["cl1", "river"],
                            ["cl1", "river"]],
        },
    })
    score = result["domain_scores"][task.target_concept]
    assert score["purity"] == pytest.approx(1 / 3, abs=1e-4)
    assert score["overall"] == pytest.approx(2 / 3, abs=1e-4)
    assert score["overall"] < 0.74


def test_single_member_clusters_are_not_perfectly_pure_and_complete():
    """Guards the metric definition, not just the arithmetic.

    An earlier version of the fix compared predicted labels to true labels as
    if they shared a namespace, which made this case score 1.0 -- every item in
    its own cluster is internally consistent, so it captures nothing. Purity
    and completeness are only meaningful against a declared concept, and a
    clustering that puts one item per cluster has completeness 1/n, not 1.
    """
    from backend.interpretability.discovery.representation_benchmark import (
        RepresentationBenchmark,
    )

    benchmark = RepresentationBenchmark()
    task = next(t for t in benchmark.tasks if t.domain == "Geography")
    result = benchmark.run_benchmark({
        task.target_concept: {
            "concept": "capital",
            "assignments": [["a", "capital"], ["b", "capital"],
                            ["c", "capital"], ["d", "capital"]],
        },
    })
    score = result["domain_scores"][task.target_concept]
    assert score["purity"] == pytest.approx(1.0)
    assert score["completeness"] == pytest.approx(0.25), (
        "one item per cluster captures exactly one of four; reporting 1.0 here "
        "means the metric is comparing label spaces instead of measuring")


def test_unmeasured_tasks_are_reported_not_silently_averaged_in():
    from backend.interpretability.discovery.representation_benchmark import (
        RepresentationBenchmark,
    )

    benchmark = RepresentationBenchmark()
    task = next(t for t in benchmark.tasks if t.domain == "Geography")
    result = benchmark.run_benchmark({
        task.target_concept: {"concept": "capital", "assignments": [["a", "capital"]]},
    })

    assert result["tasks_measured"] == 1
    assert set(result["unmeasured"]) == {
        t.target_concept for t in benchmark.tasks if t is not task
    }
    assert result["overall_rigor"] == result["domain_scores"][
        task.target_concept]["overall"], (
        "the average must be over measured tasks only")


@pytest.mark.parametrize("entry", [
    "not-a-mapping",
    {"assignments": [["a", "capital"]]},              # no concept declared
    {"concept": "capital", "assignments": "nonsense"},
    {"concept": "capital", "assignments": ["not-a-pair"]},
    {"concept": "capital", "assignments": [["a", "river"]]},  # concept absent
])
def test_unusable_assignments_are_named_rather_than_coerced(entry):
    from backend.interpretability.discovery.representation_benchmark import (
        RepresentationBenchmark,
    )

    benchmark = RepresentationBenchmark()
    task = next(t for t in benchmark.tasks if t.domain == "Geography")
    result = benchmark.run_benchmark({task.target_concept: entry})
    assert result["provenance"] == "unavailable"
    assert result["domain_scores"] == {}
    assert task.target_concept in result["unmeasured"]
    assert result["unmeasured"][task.target_concept], "the reason must be recorded"


_BENCH_PROBE = (
    "import json, sys; sys.path.insert(0, sys.argv[1]);"
    "from backend.interpretability.discovery.representation_benchmark import "
    "RepresentationBenchmark as R;"
    "b = R();"
    "t = next(x for x in b.tasks if x.domain == 'Geography');"
    "print(json.dumps(b.run_benchmark({t.target_concept: "
    "{'concept': 'capital', 'assignments': [['a','capital'],['b','river']]}}),"
    " sort_keys=True))"
)


def test_the_benchmark_does_not_depend_on_the_hash_seed():
    """The old implementation failed here, and only here.

    Within one process `hash()` is stable, so the fabricated scores looked
    reproducible to any test that ran them twice. Across processes they moved.
    """
    outputs = _under_seeds(_BENCH_PROBE, str(ROOT))
    assert len(set(outputs)) == 1, (
        "the benchmark's output depends on PYTHONHASHSEED, which means its "
        "numbers come from string hashing rather than from the assignments")


# ── The mislabelled checksum ─────────────────────────────────────────────────

_CHECKSUM_PROBE = (
    "import sys; sys.path.insert(0, sys.argv[1]);"
    "from backend.mech_platform.services.provenance_service import "
    "ProvenanceService;"
    "print(ProvenanceService().record_provenance('exp-1')['checksum'])"
)


def test_checksum_is_a_real_sha256_and_not_eight_hex_characters():
    from backend.mech_platform.services.provenance_service import ProvenanceService

    record = ProvenanceService().record_provenance("exp-1")
    digest = record["checksum"]
    assert len(digest) == 64, (
        f"checksum is {len(digest)} characters; the old value was a truncated "
        f"Python hash wearing a sha256_ prefix")
    assert all(c in "0123456789abcdef" for c in digest), digest
    assert record["checksum_algorithm"] == "sha256"
    assert not digest.startswith("sha256_"), (
        "the algorithm prefix moved into the value, where a consumer doing "
        "len(checksum) == 64 would now fail")


def test_checksum_is_stable_for_the_same_facts_and_changes_with_them():
    from backend.mech_platform.services.provenance_service import ProvenanceService

    service = ProvenanceService()
    base = service.record_provenance("exp-1", pipeline_version="2.0.0")
    same = service.record_provenance("exp-1", pipeline_version="2.0.0")
    assert same["checksum"] == base["checksum"]

    # Vary one field at a time through the keyword, never by passing a
    # positional and the same name again.
    for field in ("experiment_id", "pipeline_version", "model_version",
                  "dataset_version"):
        kwargs = dict(experiment_id="exp-1", pipeline_version="2.0.0",
                      model_version="GPT-2 Small", dataset_version="OpenWebText v1")
        kwargs[field] = kwargs[field] + "-changed"
        assert service.record_provenance(**kwargs)["checksum"] != base["checksum"], field


def test_checksum_is_stable_across_processes():
    """`hash()` is salted per process, so the old value differed every run."""
    outputs = _under_seeds(_CHECKSUM_PROBE, str(ROOT))
    assert len(set(outputs)) == 1, (
        "the checksum changed with PYTHONHASHSEED, so it was derived from "
        "Python's salted hash rather than from the values it covers")
    assert len(outputs[0]) == 64


def test_the_duplicate_service_delegates_rather_than_diverging():
    """One authority per concern: the research_platform copy is a re-export."""
    from backend.mech_platform.services import provenance_service as mech
    from backend.research_platform.services import provenance_service as research

    assert research.ProvenanceService is mech.ProvenanceService, (
        "there are two ProvenanceService implementations again; a provenance "
        "rule fixed in one can be bypassed through the other")


# ── The unsafe checkpoint load ───────────────────────────────────────────────

def test_the_sae_loader_never_disables_weights_only():
    """Structural, and checked on the AST rather than by grepping prose.

    A text search would match this very test file, and previously would have
    matched the docstring explaining why `weights_only=False` was wrong. The
    check is on the keyword arguments of `torch.load` calls, in the loader only.
    """
    loader = ROOT / "backend" / "interpretability" / "sae" / "loader.py"
    tree = ast.parse(loader.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "load"):
            continue
        if ast.unparse(func.value) != "torch":
            continue
        for keyword in node.keywords:
            if keyword.arg == "weights_only":
                value = ast.unparse(keyword.value)
                if value != "True":
                    offenders.append(f"line {node.lineno}: weights_only={value}")

    assert offenders == [], (
        "a torch.load call disables the safe unpickler: " + "; ".join(offenders))