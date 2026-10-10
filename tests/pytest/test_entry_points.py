"""An advertised entry point must be able to run, and must not overwrite provenance.

The rule under test
-------------------
**A command the project advertises must execute, and it must not write into a
record whose provenance it cannot establish.**

Two defects motivated this file, both of the same shape -- a public entry point
that looked real and was not.

1. `backend/science/benchmarking_orchestrator.py` imported four classes from
   `backend.reproductions`, a package that does not exist, and none of whose
   classes exist anywhere in the repo:

       from ..reproductions.ioi_reproduction import IOIReproduction
       from ..reproductions.induction_heads import InductionHeadsReproduction
       from ..reproductions.sae_reproduction import SparseAutoencoderReproduction
       from ..reproductions.acdc_reproduction import ACDCReproduction

   The imports were function-local, so `import backend.science` and
   `import run_benchmarks` both succeeded. Only *running* the campaign raised
   `ModuleNotFoundError`, on the first line of the loop. A deferred import error
   reads as a working entry point right up until someone depends on it.

   Worse, `_record_result` appended rows shaped
   `{"timestamp", "model", "benchmark", "reproduction_successful",
   "quality_score", "p_value", "effect_size"}` into
   `backend/science/benchmark_database.json`, a file in which every record is
   stamped `"measured": false, "provenance": "reference",
   "publication_eligible": false`. The appended rows carried none of those
   markers, so a crashed or empty run would have left entries indistinguishable
   from hand-typed fixtures. Its model list -- gpt2-small, gpt2-medium,
   gemma-2b, llama-3-8b, qwen-7b -- was also fiction; four of the five cannot be
   served here.

   Deleted rather than rewired. `BenchmarkRunner` in
   `backend/science/reproducibility` already runs the six real pipelines,
   distinguishes `NOT_RUN` from `ERROR`, and carries per-sample traces forward.

2. `run_benchmarks.py` documented itself as follows:

       # Running the campaign generates real entries in the benchmark database
       # and produces the statistical traces required for papers.

   It generated nothing, because the campaign could not start.

What is left, and why
---------------------
**Nothing.** `EXPECTED_UNRESOLVED` is empty, so any import in the repository that
names a module which does not exist is a test failure.

Three were tolerated while they were being fixed, and each was removed the moment
the fix landed:

  * `backend/benchmarking/kg_integrator.py` -> `..benchmarks.benchmark_tasks`.
    The package is `backend/benchmarking`, so the sibling module is
    `.benchmark_tasks`.
  * `frontend/scripts/reproduce_everything.py` -> `backend.benchmarks.*`. Same
    rename, from the other direction.
  * `backend/main.py` -> `backend.api.runtime_api`. This one is *not* a defect and
    stays on a separate list: `/api/v2` is an absent capability, and `main.py`
    checks `importlib.util.find_spec` before importing it, logging "capability
    absent, not degraded" and skipping the mount. A guarded optional import is the
    correct pattern for a feature that is not there; the test below pins the
    guard so the import cannot lose it.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Directories that are not part of the shipped package.
SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", ".claude", "release",
    "site-packages", ".venv", ".pytest-tmp", ".agents",
    "MECH-standalone",  # duplicate/standalone copy, not part of shipped package
}

#: Top-level names that are this project's own, so their absence is a real defect
#: rather than an optional third-party import.
LOCAL_TOP = {
    "backend", "science", "agents", "api", "core", "services", "benchmarking",
    "interpretability", "research_platform", "validation", "storage",
    "research_datasets", "sdk", "frontend",
}

#: Imports that cannot resolve, with the reason each is tolerated.
#:
#: Empty. Both entries this list held have been fixed, and in both cases
#: `test_the_declared_exceptions_are_still_real` failed the moment the fix landed,
#: which is that test doing its job -- an exception list that can only grow is a
#: permission slip, not a guard.
#:
#:   * `backend/benchmarking/kg_integrator.py` wanted `.benchmark_tasks`, not
#:     `..benchmarks.benchmark_tasks`.
#:   * `frontend/scripts/reproduce_everything.py` wanted
#:     `backend.benchmarking.*`, not `backend.benchmarks.*`.
EXPECTED_UNRESOLVED: dict[str, str] = {}

#: A module imported on purpose, behind a `find_spec` guard, for a capability that
#: is absent. Not a defect: an optional route that is not mounted must not stop
#: the app from starting. `test_runtime_api_import_stays_guarded` asserts the
#: guard is present, so this entry cannot quietly become an unguarded crash.
EXPECTED_GUARDED_OPTIONAL: dict[str, str] = {
    "backend/main.py": (
        "backend.api.runtime_api does not exist; /api/v2 is an unmounted "
        "optional capability and main.py checks find_spec before importing it"
    ),
}


def _module_exists(dotted: str) -> bool:
    """Whether a dotted name resolves to a file in this repository.

    Both the repository root and `backend/` are import roots, because the test
    invocation puts both on `sys.path` (`PYTHONPATH="backend;."`). That is why
    tests may legitimately write `science.reproducibility.ioi_pipeline` while
    application code writes `backend.science.reproducibility.ioi_pipeline`.
    """
    parts = dotted.split(".")
    for base in (ROOT, ROOT / "backend"):
        candidate = base.joinpath(*parts)
        if candidate.with_suffix(".py").exists():
            return True
        if (candidate / "__init__.py").exists():
            return True
    return False


def _unresolved_imports() -> List[Tuple[str, int, str]]:
    """Every import in the repo that names a local module which does not exist.

    AST-based rather than textual, and relative imports are resolved against the
    importing file's package -- so `from ..reproductions.ioi_reproduction import X`
    inside `backend/science/` is correctly reported as
    `backend.reproductions.ioi_reproduction`.

    The standard library is excluded via `sys.stdlib_module_names`, because an
    earlier hand-rolled allowlist of "known good" names shadowed the stdlib
    `statistics` module and reported three files as broken when they were fine.
    """
    stdlib = set(sys.stdlib_module_names)
    found: List[Tuple[str, int, str]] = []

    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if SKIP_DIRS & set(relative.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue

        parts = list(relative.with_suffix("").parts)
        is_init = bool(parts) and parts[-1] == "__init__"
        if is_init:
            parts = parts[:-1]
        package = parts if is_init else parts[:-1]

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    # level 1 == this package, level 2 == parent, and so on.
                    prefix = (package[:len(package) - (node.level - 1)]
                              if node.level > 1 else package)
                    dotted = ".".join(list(prefix)
                                      + ([node.module] if node.module else []))
                else:
                    dotted = node.module or ""
                    top = dotted.split(".")[0]
                    if not dotted or top in stdlib or top not in LOCAL_TOP:
                        continue
                if dotted and not _module_exists(dotted):
                    found.append((str(relative).replace("\\", "/"),
                                  node.lineno, dotted))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (alias.name.split(".")[0] in LOCAL_TOP
                            and not _module_exists(alias.name)):
                        found.append((str(relative).replace("\\", "/"),
                                      node.lineno, alias.name))
    return found


# ── the guard ──────────────────────────────────────────────────────────────

def test_no_unannounced_unresolvable_imports():
    """The repo may only contain the unresolvable imports it has declared."""
    tolerated = set(EXPECTED_UNRESOLVED) | set(EXPECTED_GUARDED_OPTIONAL)

    by_file: dict[str, List[str]] = {}
    for filename, lineno, dotted in _unresolved_imports():
        by_file.setdefault(filename, []).append(f"{dotted} (line {lineno})")

    unexpected = {name: mods for name, mods in by_file.items()
                  if name not in tolerated}
    assert not unexpected, (
        "imports of modules that do not exist, which make the importing file "
        "fail when reached:\n"
        + "\n".join(f"  {name}: {', '.join(mods)}"
                    for name, mods in sorted(unexpected.items()))
        + "\n\nAn import that cannot resolve is not an entry point. Either "
          "repoint it at the module that exists, or delete the caller."
    )


def test_the_declared_exceptions_are_still_real():
    """Each tolerated unresolved import must still be present and still be broken.

    Without this, fixing `backend/benchmarking/kg_integrator.py` would leave a
    stale entry in `EXPECTED_UNRESOLVED` and the exception list would rot into a
    permanent permission slip.
    """
    present = {name for name, _, _ in _unresolved_imports()}
    for filename, reason in {**EXPECTED_UNRESOLVED,
                             **EXPECTED_GUARDED_OPTIONAL}.items():
        if filename in present:
            continue
        pytest.fail(
            f"{filename} no longer has an unresolvable import, so its entry in "
            f"the tolerated list is stale and should be deleted. Reason it was "
            f"tolerated: {reason}"
        )


def test_the_reproductions_package_really_is_gone():
    """The deleted orchestrator's import target must not quietly reappear.

    If `backend/reproductions/` is ever added, the honest question is whether the
    classes in it are real measurements -- not whether the old campaign can be
    restored. This test exists so that adding the package is a deliberate act
    that trips a reviewer's attention.
    """
    assert not (ROOT / "backend" / "reproductions").exists(), (
        "backend/reproductions/ exists again. BenchmarkingOrchestrator was "
        "deleted rather than rewired because it appended unprovenanced rows into "
        "a labelled fixture. If the package is being rebuilt, build it under "
        "backend/science/reproducibility/ alongside the real pipelines."
    )


def test_runtime_api_import_stays_guarded():
    """`backend/main.py` imports a module that does not exist -- on purpose.

    `/api/v2` is an absent capability, and the honest way to say so is to check
    with `importlib.util.find_spec`, log "capability absent, not degraded", and
    skip the mount. An unguarded import would make the whole app fail to start
    over a route that was never there.
    """
    import importlib.util

    assert not (ROOT / "backend" / "api" / "runtime_api.py").exists(), (
        "backend/api/runtime_api.py now exists, so main.py's find_spec guard and "
        "its 'capability absent' warning are stale -- the router should be "
        "mounted unconditionally."
    )

    source = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")
    assert 'find_spec("backend.api.runtime_api")' in source, (
        "the optional /api/v2 import must be guarded by find_spec")
    assert importlib.util.find_spec("backend.api.runtime_api") is None

    # The import must sit in the `else` of that check, not before it.
    guard = source.index('find_spec("backend.api.runtime_api")')
    guarded_import = source.index("from backend.api.runtime_api import router")
    assert guarded_import > guard, (
        "the runtime_api import must come after the find_spec check")


# ── backend.science ────────────────────────────────────────────────────────

def test_backend_science_imports_without_the_deleted_orchestrator():
    import backend.science as science

    assert not hasattr(science, "BenchmarkingOrchestrator"), (
        "BenchmarkingOrchestrator was deleted; re-exporting it would restore a "
        "name that resolves to nothing")
    assert "BenchmarkingOrchestrator" not in science.__all__
    for name in science.__all__:
        assert hasattr(science, name), f"__all__ advertises missing {name}"


def test_the_deleted_orchestrator_is_not_referenced_anywhere():
    """No surviving file may still route through the deleted orchestrator."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if SKIP_DIRS & set(relative.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: List[str] = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, ast.Name):
                names = [node.id]
            for name in names:
                if "benchmarking_orchestrator" in name or name == "BenchmarkingOrchestrator":
                    offenders.append(f"{relative}:{node.lineno}: {name}")
    assert not offenders, (
        "references to the deleted orchestrator: " + "; ".join(offenders))


# ── run_benchmarks.py ──────────────────────────────────────────────────────

def test_run_benchmarks_is_importable_and_self_describing():
    """The replacement entry point must load and must describe itself honestly."""
    import run_benchmarks

    assert callable(run_benchmarks.main)
    assert run_benchmarks.DEFAULT_MODEL == "gpt2", (
        "the default must be the one model this repository has verified, not a "
        "list of models it cannot serve")

    doc = run_benchmarks.__doc__ or ""
    assert "benchmark_database.json" in doc, (
        "the entry point should state that it leaves the fixture alone")


def test_run_benchmarks_does_not_write_the_fixture():
    """It must not append to, or rewrite, the hand-written fixture database.

    Checked on the AST. A textual search is wrong twice over: the docstring
    legitimately explains at length why the file must not be written, and so does
    the summary it prints at the end of a run. This has now been the fourth guard
    in this effort to match the *explanation* of a fix instead of the defect.

    The invariant is that the fixture is never *opened*, not that its name is
    never mentioned.
    """
    import run_benchmarks

    tree = ast.parse(Path(run_benchmarks.__file__).read_text(encoding="utf-8"))

    opened: List[str] = []
    literal_paths: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None)
        if name != "open" or not node.args:
            continue
        target = node.args[0]
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            opened.append(repr(target.value))
            literal_paths.append(target.value)
        else:
            opened.append("<caller-supplied path>")

    assert not any("benchmark_database" in path for path in literal_paths), (
        f"run_benchmarks opens the fixture database: {literal_paths}. That file "
        f"is a labelled fixture; every record in it says measured=false and "
        f"publication_eligible=false. Measured results belong in signed "
        f"artifacts, not appended to a reference list."
    )

    # The only file this script opens is the one the caller explicitly names.
    assert opened == ["<caller-supplied path>"], (
        f"expected exactly one open(), taking the caller's --json path; found "
        f"{opened}")
    source = Path(run_benchmarks.__file__).read_text(encoding="utf-8")
    assert "open(args.json" in source
    assert "open(self.database_path" not in source


def test_run_benchmarks_makes_no_claim_it_cannot_meet():
    """The old comment claimed real entries and paper-grade traces. It made none."""
    from tests.pytest.source_assert import executable_source

    import run_benchmarks

    body = executable_source(run_benchmarks)
    for phrase in (
        "generates real entries",
        "required for papers",
        "statistical traces required",
    ):
        assert phrase not in body, (
            f"run_benchmarks still claims {phrase!r}; it runs the real "
            f"pipelines but does not write the benchmark database")


def test_run_benchmarks_refuses_a_model_it_cannot_serve(capsys):
    """The lesson from the provenance fix, applied one layer up.

    `gpt2_engine` loads one checkpoint and echoes the caller's `model_name`, so a
    request for an absent model used to return a live-looking result attributed to
    a model that never ran. This entry point must refuse instead of trying and
    hoping.
    """
    import run_benchmarks

    available, reason = run_benchmarks._weights_available_locally(
        "this-model-does-not-exist-in-any-cache")
    assert available is False
    assert "no local weights" in reason


def test_run_benchmarks_accepts_the_verified_model():
    """The refusal must not be a blanket refusal -- gpt2 is present."""
    pytest.importorskip("transformers")
    import run_benchmarks

    available, reason = run_benchmarks._weights_available_locally("gpt2")
    if not available:
        pytest.skip(f"gpt2 not in the local cache here: {reason}")


def test_run_benchmarks_refusal_returns_nonzero(capsys):
    """A mistyped --model must exit non-zero, not produce a report."""
    import run_benchmarks

    code = run_benchmarks.main(["--model", "definitely-not-a-real-model-xyz"])
    assert code == 1
    assert "REFUSING TO RUN" in capsys.readouterr().out


def test_summary_distinguishes_not_run_from_error():
    """`NOT_RUN` is a statement about scope; `ERROR` is a statement about code.

    Collapsing them loses what a reader needs to know, which is why the report
    renderer's old `PASS if fidelity > 90 else WARN` was wrong.
    """
    import run_benchmarks

    payload = {"reports": {
        "ioi": {"status": "PASS"},
        "sae": {"status": "NOT_RUN"},
        "acdc": {"status": "ERROR"},
        "arithmetic": {"status": "FIXTURE"},
        "copy_task": {"status": None},
    }}
    counts, unmeasured = run_benchmarks._summarise(payload)

    assert counts == {"PASS": 1, "NOT_RUN": 1, "ERROR": 1, "FIXTURE": 1, "UNKNOWN": 1}
    assert sorted(unmeasured) == [
        "acdc (ERROR)", "arithmetic (FIXTURE)", "copy_task (UNKNOWN)", "sae (NOT_RUN)",
    ]
    # PASS is the only status that counts as measured.
    assert counts.get("PASS", 0) + counts.get("MEASURED", 0) == 1


def test_run_benchmarks_exits_zero_when_something_measured(monkeypatch, capsys):
    """The success path must return 0 -- otherwise a real run looks like a failure.

    Verified by driving `main()` with a stubbed runner rather than by reading the
    `return 0 if measured else 1` line: a piped shell invocation reported exit 1
    for a run in which one pipeline had passed, and the contract should be pinned
    by a test rather than inferred from a shell's exit code.
    """
    import run_benchmarks

    class StubRunner:
        def __init__(self, mock_mode=False):
            self.mock_mode = mock_mode

        def run_all(self, model_id="gpt2", seed=42):
            return {"model_id": model_id, "reports": {
                "ioi": {"status": "PASS", "metrics": {"faithfulness": 0.72}},
                "sae": {"status": "NOT_RUN"},
            }}

    monkeypatch.setattr(
        "backend.science.reproducibility.benchmark_runner.BenchmarkRunner",
        StubRunner,
    )
    monkeypatch.setattr(
        run_benchmarks, "_weights_available_locally", lambda m: (True, ""))

    assert run_benchmarks.main([]) == 0
    out = capsys.readouterr().out
    assert "1 pipeline(s) produced a measurement" in out
    assert "sae (NOT_RUN)" in out, "an unmeasured benchmark must still be listed"


def test_run_benchmarks_exits_nonzero_when_only_fixtures(monkeypatch, capsys):
    """FIXTURE is not a pass. A mock-only run measures nothing and must say so."""

    class FixtureOnlyRunner:
        def __init__(self, mock_mode=False):
            self.mock_mode = mock_mode

        def run_all(self, model_id="gpt2", seed=42):
            return {"reports": {"ioi": {"status": "FIXTURE"},
                                "sae": {"status": "NOT_RUN"}}}

    import run_benchmarks

    monkeypatch.setattr(
        "backend.science.reproducibility.benchmark_runner.BenchmarkRunner",
        FixtureOnlyRunner,
    )
    monkeypatch.setattr(
        run_benchmarks, "_weights_available_locally", lambda m: (True, ""))

    assert run_benchmarks.main([]) == 1
    out = capsys.readouterr().out
    assert "No pipeline produced a measurement" in out


# ── the fixture itself ─────────────────────────────────────────────────────

def test_benchmark_database_is_still_labelled_unmeasured():
    """The fixture must stay honest after the writer was removed."""
    path = ROOT / "backend" / "science" / "benchmark_database.json"
    records = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(records, list) and records, "shape changed unexpectedly"
    for record in records:
        assert record.get("fixture") is True, record
        assert record.get("measured") is False, record
        assert record.get("provenance") == "reference", record
        assert record.get("publication_eligible") is False, record
        assert record.get("validation_eligible") is False, record
        assert "NOT A RESULT" in record.get("fixture_notice", ""), record

    # Every record names a model that cannot actually be served here, which is
    # precisely why no writer may append to this file.
    unservable = {"gemma-2b", "llama-3-8b", "qwen-7b"}
    assert {r["model"] for r in records} & unservable, (
        "expected the fixture to contain records for models that cannot run; if "
        "that changed, revisit why nothing writes this file")
