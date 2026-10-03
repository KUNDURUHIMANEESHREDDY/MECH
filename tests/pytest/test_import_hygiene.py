"""No module shadowing when backend/ is on sys.path.

The desktop app runs `python backend/main.py` with
PYTHONPATH="<repo>;<repo>/backend" (electron/main.js). That puts backend/ on
sys.path, so any top-level package name under backend/ that collides with a
third-party distribution wins over it.

It did. backend/datasets/ shadowed HuggingFace's `datasets`, so transformer_lens
failed at:

    from datasets.arrow_dataset import Dataset
    -> ModuleNotFoundError: No module named 'datasets.arrow_dataset'

and neuron inspection -- the platform's headline feature -- was dead in the
packaged desktop app while working fine when the backend was launched from the
repo root. Now renamed to backend/research_datasets.

This test is the guard. It deliberately reproduces the app's sys.path rather
than the test harness's, because the harness and the app disagree here and the
harness is the one that hides the bug.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BACKEND_DIR = REPO_ROOT / "backend"

# Distributions the app imports directly or transitively. A same-named
# directory under backend/ shadows all of them whenever backend/ is on the path.
SHADOW_RISK = [
    "datasets",     # HuggingFace, imported by transformer_lens
    "transformers",
    "torch",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",
    "pydantic",
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "requests",
    "httpx",
    "tqdm",
    "yaml",
]


@pytest.mark.parametrize("name", SHADOW_RISK)
def test_backend_does_not_shadow_third_party_package(name):
    """A same-named backend/ directory would win over site-packages."""
    collision = BACKEND_DIR / name
    if not collision.exists():
        return  # nothing to shadow with
    if not (collision / "__init__.py").exists() and not collision.is_dir():
        return
    pytest.fail(
        f"backend/{name} shadows the third-party '{name}' distribution. "
        f"electron/main.js puts backend/ on PYTHONPATH, so this directory wins "
        f"and breaks any import of '{name}'. Rename it."
    )


def test_transformer_lens_imports_under_the_apps_sys_path():
    """End-to-end: the exact sys.path the packaged desktop app uses."""
    if not _has("transformer_lens"):
        pytest.skip("transformer-lens is not installed")

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(BACKEND_DIR)!r})\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from backend.neuron_inspector import GPT2Model\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        "neuron inspection fails under the desktop app's sys.path "
        "(repo root + backend/). This is the path electron/main.js uses.\n"
        f"stderr:\n{result.stderr.strip()[-900:]}"
    )


def test_no_backend_package_name_collides_with_an_installed_distribution():
    """Catch the general case: backend/ dirs named after installed packages.

    A backend/ directory is only a problem when a *distribution of the same name
    is installed*, because then backend/ wins the import. `backend/api` and
    `backend/agents` are fine -- nothing installs those names.
    """
    from importlib import metadata

    # Top-level import names provided by installed distributions.
    installed = set()
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").strip().lower().replace("-", "_")
        if name:
            installed.add(name)
    # A distribution can also ship submodules; catch the obvious top-level ones.
    try:
        import torch  # noqa: F401
        installed.add("torch")
    except ImportError:
        pass

    offenders = []
    for entry in sorted(BACKEND_DIR.iterdir()):
        if not entry.is_dir() or entry.name == "__pycache__":
            continue
        if not (entry / "__init__.py").exists():
            continue
        if entry.name.lower().replace("-", "_") in installed:
            offenders.append(entry.name)

    assert not offenders, (
        f"backend/ directories shadow installed distributions: {offenders}. "
        f"electron/main.js puts backend/ on PYTHONPATH, so these win. "
        f"Rename them."
    )


def _has(module: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False