"""Dependency-drift gate for the interpretability stack.

The neuron-inspection path silently stopped working because requirements.txt
allowed transformer-lens>=2.0.0, which now resolves to 4.0, where
HookedTransformer was removed. Nothing in the suite imported the stack, and the
one test that touched it accepted an ImportError as a pass.

This closes that gap. The rule it encodes:

* if the ML stack is NOT installed, skip -- a bare dev environment legitimately
  lacks torch, and CI installs it explicitly;
* if it IS installed, importing it MUST work. A half-installed or
  API-incompatible stack is a failure, not a skip.

That distinction is the whole point. A skip-on-any-error version of this test
would have passed while the feature was broken.
"""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _installed(name: str) -> bool:
    """True only if the module can actually be located, not merely imported."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


requires_ml = pytest.mark.skipif(
    not _installed("transformer_lens"),
    reason="transformer-lens is not installed; nothing to gate",
)


def test_transformer_lens_exposes_hooked_transformer():
    """The symbol the codebase actually calls must exist."""
    if not _installed("transformer_lens"):
        pytest.skip("transformer-lens is not installed")

    # transformer_lens 4.x raises a helpful AttributeError lazily; surface it
    # as a plain assertion failure so the message names the real problem.
    result = subprocess.run(
        [sys.executable, "-c",
         "import transformer_lens as tl; tl.HookedTransformer; print('ok')"],
        capture_output=True, text=True, timeout=180, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        "transformer_lens does not expose HookedTransformer. "
        "backend/ uses the 2.x API and requirements.txt pins <3.0.0; if this "
        "fired, the pin was removed or bypassed.\n"
        f"stderr: {result.stderr.strip()[-500:]}"
    )


@requires_ml
def test_gpt2_model_module_imports():
    """backend.interpretability.gpt2_model is the entry to neuron inspection.

    Runs in a subprocess from the repo root, not in-process. Under pytest,
    conftest.py puts backend/ on sys.path, which is not how the app runs and
    hides the module-shadowing class of bug entirely. See
    test_import_hygiene.py, which asserts the app's real sys.path.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "from backend.interpretability.gpt2_model import GPT2Model; print('ok')"],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"gpt2_model cannot import. stderr:\n{result.stderr.strip()[-800:]}"
    )


def test_neuron_inspector_imports():
    """backend.neuron_inspector re-exports GPT2Model for the sidecar."""
    if not _installed("transformer_lens"):
        pytest.skip("transformer-lens is not installed")
    result = subprocess.run(
        [sys.executable, "-c",
         "from backend.neuron_inspector import GPT2Model; print('ok')"],
        capture_output=True, text=True, timeout=180, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"neuron inspection cannot import. stderr:\n{result.stderr.strip()[-800:]}"
    )


# ---------------------------------------------------------------------------
# The pin itself
# ---------------------------------------------------------------------------


def test_requirements_pin_transformer_lens_below_4():
    """A regression guard on the pin, not on the installed version.

    Reading requirements.txt rather than the live environment means this fails
    in CI even on a machine that happens to have a working install.
    """
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = [
        l.strip() for l in text.splitlines()
        if l.strip() and not l.strip().startswith("#")
        and "transformer-lens" in l
    ]
    assert lines, "transformer-lens is missing from requirements.txt"
    spec = lines[0]

    # Parse the upper bound rather than grepping for a literal. "<3.0",
    # "<3.0.0" and "<3" are the same constraint; the original assertion
    # rejected two of the three on formatting alone, so a correct, stricter pin
    # could fail this test.
    upper = re.search(r"<\s*([0-9]+(?:\.[0-9]+)*)", spec)
    assert upper, f"no upper bound in {spec!r}; it is unbounded"
    assert float(upper.group(1)) < 4.0, (
        f"transformer-lens must be capped below 4.0, got: {spec!r}. "
        f"4.0 removed HookedTransformer, which backend/ depends on."
    )


def test_requirements_pin_transformers_floor_matches_transformer_lens():
    """transformer-lens 2.x needs transformers>=4.43; keep the floors coherent."""
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    floor = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("transformers"):
            # Form is "transformers>=4.43.0" or "transformers>=4.43,<5".
            spec = line[len("transformers"):]
            if spec.startswith(">="):
                floor = spec[2:].split(",")[0].strip()
    assert floor is not None, "no transformers floor in requirements.txt"
    assert tuple(int(p) for p in floor.split(".")[:2]) >= (4, 43), (
        f"transformers floor is {floor}; transformer-lens 2.x requires >=4.43"
    )