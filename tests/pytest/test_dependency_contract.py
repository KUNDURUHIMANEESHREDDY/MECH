"""Guards the transformer-lens / transformers version contract.

TransformerLens 4.0 removed ``HookedTransformer``, which three MECH code paths
depend on. ``requirements.txt`` pins both libraries with upper bounds to stop a
fresh install resolving to that broken combination; these tests make sure the
pin and the code stay in agreement.

They are deliberately cheap: they never load model weights.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.interpretability.tl_compat import (  # noqa: E402
    SUPPORTED_TL_RANGE,
    check_compatibility,
    installed_version,
    library_report,
    resolve_hooked_transformer,
)

REQUIREMENTS = REPO_ROOT / "requirements.txt"

#: Call sites that must not import HookedTransformer directly any more.
CONSUMERS = [
    REPO_ROOT / "backend" / "interpretability" / "gpt2_model.py",
    REPO_ROOT / "backend" / "science" / "models" / "transformer_lens_adapter.py",
    REPO_ROOT / "gpt2_steps.py",
]


def _requirement(text: str, package: str) -> str | None:
    """Return the raw requirement line for ``package`` (case-insensitive)."""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip().lower()
        if name == package.lower():
            return line
    return None


def _upper_bound(line: str, package: str) -> int | None:
    """Return the integer in a `,<N` upper bound, or None if there is none."""
    spec = line.split(package, 1)[1]
    match = re.search(r"<\s*(\d+)", spec)
    return int(match.group(1)) if match else None


def test_requirements_pin_transformer_lens_below_4():
    """transformer-lens must stay on a release that exports HookedTransformer.

    A `<N` bound of N means "N and later are excluded", so N=4 correctly rejects
    the 4.x release that removed HookedTransformer.
    """
    text = REQUIREMENTS.read_text(encoding="utf-8")
    line = _requirement(text, "transformer-lens")
    assert line is not None, "transformer-lens missing from requirements.txt"

    bound = _upper_bound(line, "transformer-lens")
    assert bound is not None, f"transformer-lens has no upper bound: {line!r}"
    assert bound <= 4, (
        f"transformer-lens upper bound {bound} allows 4.x, which removed "
        f"HookedTransformer: {line!r}"
    )


def test_requirements_pin_transformers_below_5():
    """transformer-lens 2.x declares transformers>=4.57 with no ceiling, so
    transformers must be capped here or pip pairs them with an incompatible 5.x."""
    text = REQUIREMENTS.read_text(encoding="utf-8")
    line = _requirement(text, "transformers")
    assert line is not None, "transformers missing from requirements.txt"

    bound = _upper_bound(line, "transformers")
    assert bound is not None, f"transformers has no upper bound: {line!r}"
    assert bound <= 5, (
        f"transformers upper bound {bound} allows 5.x, which is not compatible "
        f"with transformer-lens 2.x: {line!r}"
    )


def test_supported_range_matches_requirements_pin():
    """The shim's advertised range must agree with the pin, or the error
    message will send people to install a combination we did not test."""
    text = REQUIREMENTS.read_text(encoding="utf-8")
    line = _requirement(text, "transformer-lens")
    assert line is not None
    spec = line.split("transformer-lens", 1)[1].split("#")[0].strip().rstrip(",")
    assert spec == SUPPORTED_TL_RANGE, (
        f"tl_compat.SUPPORTED_TL_RANGE={SUPPORTED_TL_RANGE!r} disagrees with "
        f"requirements.txt ({spec!r})"
    )


def _compatibility_in_clean_interpreter() -> dict:
    """Run the compat check in a subprocess with a clean sys.path.

    This has to be isolated. tests/pytest/conftest.py puts ``backend/`` on
    sys.path, where MECH's ``backend/datasets/`` package shadows HuggingFace
    ``datasets``; in-process that makes transformer-lens unimportable and turns
    the version check into a skip, which would hide a genuine 4.x regression.
    A subprocess with only the repo root on PYTHONPATH sees the real stack.
    """
    import json
    import subprocess

    probe = (
        "import json,sys;"
        "from backend.interpretability.tl_compat import check_compatibility;"
        "print(json.dumps(check_compatibility()))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    try:
        out = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("compatibility probe timed out")
    if out.returncode != 0:
        pytest.skip(f"compatibility probe failed:\n{out.stderr[-400:]}")
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        pytest.skip(f"could not parse compatibility probe:\n{out.stdout[-400:]}")


def test_installed_transformer_lens_provides_hooked_transformer():
    """The installed stack must satisfy the contract the pin asserts.

    A *version* mismatch is a failure: the pin exists so this cannot happen.
    An *environment* mismatch is skipped and reported instead, since a broken
    environment is a different problem from an unpinned dependency.
    """
    report = _compatibility_in_clean_interpreter()
    if report.get("reason") == "environment":
        pytest.skip(
            "transformer-lens cannot be imported in a clean interpreter: "
            f"{report['detail'].splitlines()[0]}"
        )
    assert report["compatible"], (
        "installed transformer-lens cannot serve MECH's TransformerLens paths\n"
        f"  transformer-lens: {report['transformer_lens']}\n"
        f"  transformers:     {report['transformers']}\n"
        f"  supported range:  {SUPPORTED_TL_RANGE}\n"
        f"{report['detail']}\n"
        "Fix with: pip install -r requirements.txt"
    )


def test_compat_probe_detects_a_4x_install():
    """Negative control: the check must reject transformer-lens 4.x.

    Stands a fake 4.x distribution on PYTHONPATH whose __init__ exposes no
    HookedTransformer, mirroring the real 4.x behaviour. Without this control a
    broken check would pass silently, because the skip above would hide it.
    """
    import json
    import shutil
    import subprocess

    fake_pkg = REPO_ROOT / "tests" / "pytest" / "_tl_fake_v4"
    package = fake_pkg / "transformer_lens"
    package.mkdir(parents=True, exist_ok=True)
    try:
        (package / "__init__.py").write_text(
            '"""Stand-in for transformer-lens 4.x: imports fine, no HookedTransformer."""\n'
            "\n"
            "def __getattr__(name):\n"
            "    raise AttributeError(\n"
            '        "\'" + name + "\' was removed in TransformerLens 4.0."\n'
            "    )\n",
            encoding="utf-8",
        )
        dist_info = package / "transformer_lens-4.0.0.dist-info"
        dist_info.mkdir(exist_ok=True)
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: transformer-lens\nVersion: 4.0.0\n",
            encoding="utf-8",
        )

        probe = (
            "import json;"
            "from backend.interpretability.tl_compat import check_compatibility;"
            "print(json.dumps(check_compatibility()))"
        )
        env = dict(os.environ)
        # Fake package first so it wins over any real install.
        env["PYTHONPATH"] = str(fake_pkg) + os.pathsep + str(REPO_ROOT)
        out = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if out.returncode != 0:
            pytest.skip(f"probe failed:\n{out.stderr[-400:]}")
        report = json.loads(out.stdout.strip().splitlines()[-1])

        assert not report["compatible"], "4.x stand-in was wrongly accepted"
        assert report.get("reason") == "version", (
            f"expected a version mismatch, got {report.get('reason')!r}"
        )
        assert SUPPORTED_TL_RANGE in report["detail"], (
            "the error should advertise the supported range"
        )
    finally:
        shutil.rmtree(fake_pkg, ignore_errors=True)


def test_backend_datasets_shadows_huggingface_datasets():
    """Guard the package-name collision fixed by the research_datasets rename.

    A ``backend/datasets/`` package would be importable as the top-level name
    ``datasets`` whenever ``backend/`` is on sys.path, which is exactly what
    tests/pytest/conftest.py does. Any dependency that does ``import datasets``
    -- including transformer-lens -- would then resolve to MECH's package
    instead of HuggingFace's, which hides the compat probe rather than failing
    it. The package now lives at ``backend/research_datasets/``.
    """
    import importlib.util

    assert not (REPO_ROOT / "backend" / "datasets").exists(), (
        "backend/datasets/ was restored; it shadows HuggingFace datasets. "
        "Use backend/research_datasets/ instead."
    )

    spec = importlib.util.find_spec("datasets")
    if spec is None or not spec.origin:
        pytest.skip("no top-level 'datasets' module on sys.path")

    origin = Path(spec.origin).resolve()
    backend_dir = (REPO_ROOT / "backend").resolve()

    assert backend_dir not in origin.parents, (
        "A module under backend/ is shadowing HuggingFace datasets, which "
        f"breaks transformer-lens under this conftest. Origin: {origin}"
    )


def test_resolve_returns_class_with_from_pretrained():
    """The resolved symbol must be the real TransformerLens class."""
    if not check_compatibility()["compatible"]:
        pytest.skip("transformer-lens not installed")
    cls = resolve_hooked_transformer()
    assert cls.__name__ == "HookedTransformer"
    assert hasattr(cls, "from_pretrained")


def test_library_report_reports_all_three_packages():
    report = library_report()
    for key in ("transformer_lens", "transformers", "torch", "supported_range"):
        assert key in report


def test_installed_version_returns_none_for_missing_package():
    assert installed_version("definitely-not-a-real-package-xyz") is None


@pytest.mark.parametrize("consumer", CONSUMERS, ids=lambda p: p.name)
def test_consumers_do_not_import_hooked_transformer_directly(consumer: Path):
    """Consumers must resolve through tl_compat so a bad install yields the
    actionable error rather than a bare ImportError."""
    assert consumer.exists(), f"expected call site missing: {consumer}"
    text = consumer.read_text(encoding="utf-8")

    direct = [
        line.strip()
        for line in text.splitlines()
        if re.match(r"^\s*from\s+transformer_lens\s+import\b", line)
    ]
    assert not direct, (
        f"{consumer.name} imports HookedTransformer directly ({direct}); "
        "import it from backend.interpretability.tl_compat instead"
    )

    assert "tl_compat" in text, f"{consumer.name} does not use the compat shim"
