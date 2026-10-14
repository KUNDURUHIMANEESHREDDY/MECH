"""Mutation check: do the tests detect a deliberate break in the fixes?

Question 2 of the audit -- "can the test reliably detect a violation?" -- is
answerable only by breaking the behaviour on purpose and confirming the test
goes red. Every other row in that table is evidence *about* the code; this one
is evidence about the tests.

Four production fixes from this session are mutated in turn. Each mutation is
a one-line semantic break, each is reverted in a `finally`, and each must turn
a specific green test red. A mutation that leaves the suite green is a test
that cannot fail, which is the failure mode this project has produced three
times (argument-echo, canonical-marker suffix, prose-matched assertion).

Safety
------
Files are backed up by content and restored in `finally`. Nothing is restored
via git -- the working tree holds uncommitted work, so `git checkout` would
destroy it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Tuple

ROOT = Path(__file__).resolve().parents[2]

#: (label, relative path, original text, replacement, test node id)
MUTATIONS = [
    (
        "patch_activation range check removed",
        "backend/science/models/gpt2_adapter.py",
        "if not isinstance(neuron_index, int) or not 0 <= neuron_index < width:",
        "if not isinstance(neuron_index, int) or not 0 <= neuron_index < width "
        "and False:",
        "tests/pytest/test_activation_patch_placement.py::"
        "test_an_out_of_range_index_is_refused_not_silently_skipped",
    ),
    (
        "patch_activation writes in place again",
        "backend/science/models/gpt2_adapter.py",
        "patched = output.clone()\n                patched[0, -1, neuron_index] "
        "= patch_value\n                return patched",
        "output[0, -1, neuron_index] = patch_value\n                return output",
        "tests/pytest/test_activation_patch_placement.py::"
        "test_patch_activation_matches_an_independent_write_to_the_same_coordinate",
    ),
    (
        "model_fingerprint returns a constant",
        "backend/science/models/gpt2_adapter.py",
        "digest.update(name.encode(\"utf-8\"))",
        "return \"sha256:constant\"\n            digest.update(name.encode(\"utf-8\"))",
        "tests/pytest/test_model_identity_seal.py::"
        "test_a_mutated_weight_changes_the_fingerprint",
    ),
    (
        "causal_tracing nested markers removed",
        "backend/interpretability/causal/causal_tracing.py",
        "\"field_provenance\": {\n                    \"indirect_effect\": "
        "\"reference\",\n                    \"top_causal_component\": \"reference\",\n"
        "                },",
        "",
        "tests/pytest/test_provenance_recursion.py::"
        "test_the_causal_tracing_response_marks_every_number_it_returns",
    ),
]


@contextmanager
def mutated(relative: str, original: str, replacement: str) -> Iterator[Path]:
    """Apply a mutation for the duration of the block, then restore exactly."""
    path = ROOT / relative
    backup = path.with_suffix(path.suffix + ".mutation-backup")
    shutil.copy2(path, backup)
    try:
        source = path.read_text(encoding="utf-8")
        assert original in source, (
            f"mutation target not found in {relative}: {original[:60]!r}. The "
            f"production code has changed shape; this mutation is stale and "
            f"would prove nothing."
        )
        path.write_text(source.replace(original, replacement, 1),
                        encoding="utf-8")
        yield path
    finally:
        shutil.move(str(backup), str(path))


def _run(node: str) -> Tuple[bool, str]:
    """Run one test node. Returns (failed_as_expected, output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", node, "-q", "-p", "no:randomly",
         "--no-header", "-x"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800,
        env={**_env()},
    )
    return result.returncode != 0, (result.stdout + result.stderr)[-1200:]


def _env():
    import os
    env = dict(os.environ)
    env["MECH_SKIP_SOURCE_STABILITY_CHECK"] = "1"
    return env


def test_every_mutation_is_detected_by_its_test() -> None:
    """Each mutation, each required red.

    Reported as one test rather than four so a failure names the specific
    mutation that survived. A surviving mutation means the test it targets
    cannot detect that class of break -- which is worse than having no test,
    because it reports coverage.
    """
    survivors: list = []
    for label, relative, original, replacement, node in MUTATIONS:
        try:
            with mutated(relative, original, replacement):
                detected, output = _run(node)
        except AssertionError as exc:
            survivors.append(f"{label}: {exc}")
            continue

        if not detected:
            survivors.append(
                f"{label}\n    mutation: {replacement[:70]!r}\n"
                f"    test: {node}\n    -> PASSED with the behaviour broken"
            )

    assert not survivors, (
        f"{len(survivors)}/{len(MUTATIONS)} mutations survived; the tests "
        f"targeting them cannot detect that break:\n\n  "
        + "\n\n  ".join(survivors)
    )