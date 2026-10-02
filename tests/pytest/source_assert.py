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
from typing import Any


def executable_source(module: Any) -> str:
    """Return a module's source with docstrings stripped, via the AST.

    Only code that actually runs is preserved. A module docstring or a comment
    describing a removed formula is not executable, so it cannot be what a
    guard test is guarding against.
    """
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def doc_and_comments(module: Any) -> str:
    """Return the module's raw source, docstrings and comments included.

    Use this when the assertion is genuinely about prose -- for example that a
    rejection is documented, so the next person does not re-add it.
    """
    return inspect.getsource(module)
