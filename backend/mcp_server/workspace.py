"""Workspace containment for the MCP control plane.

Same rule as the desktop sidecar (`verify_workspace_containment.py`): every
file/shell/git path must resolve inside an allowed root.

Trust boundary (immutable): `MECH_WORKSPACE_ROOTS` (os.pathsep-separated)
plus the repo root. `register_open_root()` may only record a directory
*inside* that boundary — it can never enlarge it. An MCP caller asking to
open `/` or a sibling directory is rejected; the operator must set
`MECH_WORKSPACE_ROOTS` to include it first.

Relative paths resolve against the repo root, not the process cwd, because a
packaged backend runs with cwd set to the resources directory.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_opened_roots: set[str] = set()


def configured_roots() -> list[Path]:
    """Immutable trust boundary: env roots plus the repo root.

    Opened projects are deliberately excluded — they are grants *within*
    the boundary, never extensions of it.
    """
    raw = os.environ.get("MECH_WORKSPACE_ROOTS", "")
    roots: list[Path] = [Path(p) for p in raw.split(os.pathsep) if p.strip()]
    roots.append(REPO_ROOT)
    uniq: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved not in uniq:
            uniq.append(resolved)
    return uniq


def workspace_roots() -> list[Path]:
    roots: list[Path] = list(configured_roots())
    roots.extend(Path(p) for p in _opened_roots)
    uniq: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved not in uniq:
            uniq.append(resolved)
    return uniq


def clear_open_roots() -> None:
    """Forget explicitly opened projects (test isolation)."""
    _opened_roots.clear()


def register_open_root(path: str) -> Path:
    """Record an explicitly opened project directory as an allowed root.

    The directory must already lie inside the configured trust boundary
    (`configured_roots()`). Anything else — `/`, home, a sibling, a `..`
    escape, a symlink/junction resolving outside — is rejected so the MCP
    caller can never enlarge its own authority. Set `MECH_WORKSPACE_ROOTS`
    to include the directory first.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("project path must be a non-empty string")
    candidate = Path(path.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ValueError(f"project path cannot be resolved: {exc}") from exc
    if not resolved.is_dir():
        raise ValueError(f"project path is not a directory: {path}")
    for root in configured_roots():
        try:
            resolved.relative_to(root)
            break
        except ValueError:
            continue
    else:
        allowed = ", ".join(str(r) for r in configured_roots())
        raise ValueError(
            f"project path is outside the allowed workspace roots: {resolved}. "
            f"Allowed: {allowed}. Set MECH_WORKSPACE_ROOTS to include it."
        )
    _opened_roots.add(str(resolved))
    return resolved


def resolve_contained(path_str: str) -> Path:
    """Resolve a user-supplied path, refusing anything outside allowed roots."""
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(path_str.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ValueError(f"path cannot be resolved: {exc}") from exc
    for root in workspace_roots():
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    allowed = ", ".join(str(r) for r in workspace_roots())
    raise ValueError(
        f"path is outside the allowed workspace roots: {resolved}. "
        f"Allowed: {allowed}"
    )
