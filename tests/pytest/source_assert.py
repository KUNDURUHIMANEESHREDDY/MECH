"""Helpers for asserting on source code.

Guard tests here check that a fabricated literal has not been reintroduced.
The obvious approach -- grepping ``inspect.getsource(module)`` -- fails: the
comments and docstrings explaining *why* a formula was removed contain the very
strings the test exists to forbid. Two tests in this suite were caught by that
false positive, so the parsing lives here and is shared.
"""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
from typing import Any, Iterator, Sequence


def source_of(module_or_text: Any) -> str:
    """Return raw source text for a module, class, function, or literal string.

    Accepting a string lets a guard test check a file it has not imported -- a
    deleted script, or a path under a directory the test deliberately does not
    execute. Passing a string through to `inspect.getsource` raises
    `TypeError: ... expected, got str`, which is a confusing way to learn that.
    """
    if isinstance(module_or_text, str):
        return module_or_text
    return inspect.getsource(module_or_text)


def executable_source(module_or_text: Any) -> str:
    """Return source with docstrings stripped, via the AST.

    Only code that actually runs is preserved. A module docstring or a comment
    describing a removed formula is not executable, so it cannot be what a
    guard test is guarding against.

    Comments are also dropped, because `ast.unparse` drops them anyway. Note
    that string literals inside real expressions are *preserved*: a guard
    matching on text must account for a docstring that explains the very
    anti-pattern the guard forbids.
    """
    tree = ast.parse(source_of(module_or_text))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def doc_and_comments(module_or_text: Any) -> str:
    """Return raw source, docstrings and comments included.

    Use this when the assertion is genuinely about prose -- for example that a
    rejection is documented, so the next person does not re-add it.
    """
    return source_of(module_or_text)


#: Directory names a repository-wide source scan never needs to look inside.
#: Every one of these is generated, vendored, or a scratch copy of the tree, so
#: scanning it either finds nothing or finds a second copy of the real sources.
PRUNED_DIR_NAMES = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "site-packages", "dist-packages",
    "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    ".tox", ".venv", "venv", "env",
    "build", "dist", "release",
    ".pytest-tmp", "htmlcov", ".next",
    "MECH-standalone",  # duplicate copy of the tree, not the shipped package
})


def walk_python_files(root: Path,
                      extra_pruned: Sequence[str] = ()) -> Iterator[Path]:
    """Every ``*.py`` under *root*, without descending into excluded trees.

    ``Path.rglob`` cannot prune. It walks every directory it can reach and the
    caller filters afterwards, so the filter saves nothing: on this checkout a
    plain ``ROOT.rglob("*.py")`` walks **143 491** files, because the repo root
    carries ~300 leftover ``pip-*`` scratch directories and a staged CPython
    runtime under ``frontend/build/python``. Two tests in this suite did exactly
    that and spent over a minute each before parsing a single file.

    ``os.walk`` is used instead because it hands the caller the directory list
    *before* descending, which is the only point where the cost can be removed.

    ``extra_pruned`` adds names for one caller's own exclusions -- for example a
    build copy it must ignore. Names match a directory's basename, and any name
    starting with ``.pytest-tmp`` is treated as a prefix too, because
    ``pytest.ini``'s ``addopts`` plus a per-session basetemp leaf invent siblings
    such as ``.pytest-tmp-lead``.
    """
    pruned = set(PRUNED_DIR_NAMES) | set(extra_pruned)
    root = Path(root)
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames
            if name not in pruned and not name.startswith(".pytest-tmp")
        ]
        for filename in filenames:
            if filename.endswith(".py"):
                yield Path(current) / filename

