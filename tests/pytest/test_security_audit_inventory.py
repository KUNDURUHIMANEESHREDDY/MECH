from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "security_audit_inventory.py"
SPEC = importlib.util.spec_from_file_location("security_audit_inventory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cpu_torch_build_is_audited_as_its_upstream_release():
    content, normalized = MODULE.render_inventory([
        ("Torch", "2.14.1+cpu"),
        ("Fast_API", "0.120.0"),
    ])

    assert content == "fast-api==0.120.0\ntorch==2.14.1\n"
    assert normalized == [
        "Torch==2.14.1+cpu -> torch==2.14.1 (audit upstream release; CPU build suffix only)"
    ]


def test_unreviewed_local_build_suffix_fails_closed():
    with pytest.raises(ValueError, match="Unsupported local build label"):
        MODULE.render_inventory([("some-package", "1.2.3+vendor1")])


def test_non_cpu_torch_build_suffix_fails_closed():
    with pytest.raises(ValueError, match="Unsupported local build label"):
        MODULE.render_inventory([("torch", "2.14.1+cu126")])


def test_conflicting_duplicate_installed_versions_fail_closed():
    with pytest.raises(ValueError, match="Conflicting installed versions"):
        MODULE.render_inventory([
            ("Example_Pkg", "1.0.0"),
            ("example-pkg", "2.0.0"),
        ])


def test_inventory_order_is_deterministic():
    forward, _ = MODULE.render_inventory([("zeta", "2.0"), ("alpha", "1.0")])
    backward, _ = MODULE.render_inventory([("alpha", "1.0"), ("zeta", "2.0")])

    assert forward == backward == "alpha==1.0\nzeta==2.0\n"
