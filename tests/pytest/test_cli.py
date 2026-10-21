"""Tests for the `mech` CLI (spec §22).

What these pin:

* the console script exists in ``pyproject.toml`` and points at the CLI;
* ``doctor`` works offline and ``models list`` reports the real catalogue (or
  NOT_EXECUTED), never an invented one;
* a model this build cannot load exits non-zero with NOT_EXECUTED and no
  architecture numbers -- a refusal is not a result;
* prompt files are validated before anything executes;
* experiments come from the real ``DesktopStorage`` at ``MECH_STORAGE_DB``;
* SAE surfaces are reference metadata, and inspection is NOT_EXECUTED without a
  local checkpoint (no downloads).

Nothing here loads GPT-2 weights or touches the network.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

from backend.cli.main import main

REPO_ROOT = Path(__file__).resolve().parents[2]


def _captured(capsys):
    result = capsys.readouterr()
    return result.out, result.out + result.err


# ---------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------


def test_pyproject_declares_the_mech_console_script():
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["name"] == "mech"
    assert config["project"]["scripts"]["mech"] == "backend.cli.main:main"
    assert "dependencies" not in config["project"] or config["project"]["dependencies"] == []

    requires_python = config["project"]["requires-python"]
    assert requires_python == f">={sys.version_info.major}.{sys.version_info.minor}"
    assert sys.version_info >= (3, 11)

    # Explicit package configuration: only the backend tree, never tests/frontend.
    find = config["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["backend*"]
    assert "tests*" in find["exclude"]
    assert "frontend*" in find["exclude"]

    # pytest.ini stays authoritative; this file must not carry pytest config.
    assert "pytest" not in config.get("tool", {})


def test_the_entry_point_module_is_importable():
    import backend.cli.main as cli

    assert callable(cli.main)
    assert cli.main.__module__ == "backend.cli.main"


def test_no_subcommand_is_a_usage_error():
    assert main([]) == 2


# ---------------------------------------------------------------------------
# doctor / models
# ---------------------------------------------------------------------------


def test_doctor_runs_offline_and_exits_zero(capsys):
    assert main(["doctor"]) == 0
    out, _ = _captured(capsys)
    assert "MECH doctor" in out
    assert "storage" in out
    assert "PASS" in out


def test_doctor_json_payload_is_structured(capsys):
    assert main(["doctor", "--json"]) == 0
    out, _ = _captured(capsys)
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["offline"] is True
    assert {check["check"] for check in payload["checks"]} >= {"python", "storage"}


def test_models_list_returns_the_real_catalogue_or_not_executed(capsys):
    code = main(["models", "list", "--json"])
    out, _ = _captured(capsys)
    payload = json.loads(out)

    if code == 0:
        assert "gpt2-small" in payload["models"]
        assert payload["provenance"] == "reference"
        assert payload["small_models"], "the small-model registry must come from the service"
    else:
        assert code == 3
        assert payload["status"] == "NOT_EXECUTED"
        assert payload["reason"]


def test_models_inspect_a_model_this_build_cannot_load_is_not_executed(capsys):
    code = main(["models", "inspect", "gpt2-medium"])
    out, combined = _captured(capsys)

    assert code == 3
    assert "NOT_EXECUTED" in out
    assert "gpt2-medium" in combined
    # Nothing was loaded, so no architecture may be printed for it.
    for fabricated in ("50257", "768", "3072", "n_layers"):
        assert fabricated not in out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_forward_refuses_a_model_the_engine_does_not_serve(capsys):
    code = main(
        ["run", "forward", "--model", "gpt2-medium", "--prompt", "The capital of France is"]
    )
    out, combined = _captured(capsys)

    assert code == 3
    assert "NOT_EXECUTED" in out
    assert "gpt2-medium" in combined
    assert "top5" not in out.lower()
    assert "next_token" not in out.lower()


def test_prompt_file_is_validated_before_execution(tmp_path, capsys):
    missing = tmp_path / "missing.txt"
    code = main(["run", "logit-lens", "--prompt-file", str(missing)])
    out, combined = _captured(capsys)

    assert code == 2
    assert "does not exist" in combined
    assert "NOT_EXECUTED" not in out


def test_oversized_prompt_file_is_a_usage_error(tmp_path, capsys):
    oversized = tmp_path / "big.txt"
    oversized.write_text("x" * (2048 + 1), encoding="utf-8")

    code = main(["run", "forward", "--prompt-file", str(oversized)])
    out, combined = _captured(capsys)

    assert code == 2
    assert "exceeds 2048 characters" in combined
    assert "NOT_EXECUTED" not in out


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


def test_experiments_list_reads_the_configured_storage(tmp_path, monkeypatch, capsys):
    from backend.storage import DesktopStorage

    db_path = tmp_path / "cli-experiments.db"
    monkeypatch.setenv("MECH_STORAGE_DB", str(db_path))

    assert main(["experiments", "list", "--json"]) == 0
    out, _ = _captured(capsys)
    empty = json.loads(out)
    assert empty["count"] == 0
    assert empty["experiments"] == []
    assert empty["storage"] == str(db_path)
    assert empty["provenance"] == "none"

    store = DesktopStorage(str(db_path))
    store.initialize()
    store.add_experiment(
        {"id": "exp_cli_1", "name": "cli probe", "status": "recorded"}
    )

    assert main(["experiments", "list"]) == 0
    out, _ = _captured(capsys)
    assert "exp_cli_1" in out
    assert "cli probe" in out

    assert main(["experiments", "show", "exp_cli_1"]) == 0
    out, _ = _captured(capsys)
    assert "cli probe" in out

    assert main(["experiments", "export", "exp_cli_1", "--format", "markdown"]) == 0
    out, _ = _captured(capsys)
    assert "# Experiment `exp_cli_1`" in out
    assert "nothing was computed" in out


def test_experiments_unknown_id_is_not_found(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MECH_STORAGE_DB", str(tmp_path / "empty.db"))

    code = main(["experiments", "show", "exp_missing"])
    out, combined = _captured(capsys)

    assert code == 4
    assert "not found" in combined
    assert "NOT_EXECUTED" not in out


def test_experiments_export_rejects_an_unknown_format(tmp_path, monkeypatch):
    monkeypatch.setenv("MECH_STORAGE_DB", str(tmp_path / "empty.db"))
    assert main(["experiments", "export", "x", "--format", "pdf"]) == 2


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------


def _project_sessions(tmp_path, monkeypatch):
    """A storage authority with two projects, and an export root."""
    from backend.storage import DesktopStorage

    db_path = tmp_path / "cli-projects.db"
    monkeypatch.setenv("MECH_STORAGE_DB", str(db_path))
    monkeypatch.setenv("MECH_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.delenv("MECH_EXPORT_CONFIRM_BYTES", raising=False)
    store = DesktopStorage(db_path)
    store.initialize()
    store.add_session({"id": "s-alpha", "project_id": "alpha"})
    store.add_session({"id": "s-beta", "project_id": "beta"})
    return store


def _archives(root):
    if not root.exists():
        return []
    return sorted(root.rglob("*.interp-project"))


def test_projects_export_estimate_writes_nothing(tmp_path, monkeypatch, capsys):
    _project_sessions(tmp_path, monkeypatch)
    code = main(["projects", "export", "alpha", "--estimate"])
    out, _ = _captured(capsys)
    assert code == 0
    assert "estimated_bytes" in out
    assert "sessions_included" in out
    assert _archives(tmp_path / "exports") == []


def test_projects_export_writes_the_archive(tmp_path, monkeypatch, capsys):
    _project_sessions(tmp_path, monkeypatch)
    code = main(["projects", "export", "alpha"])
    out, _ = _captured(capsys)
    assert code == 0
    assert (tmp_path / "exports" / "alpha.interp-project").is_file()
    assert "alpha.interp-project" in out
    assert "bytes_written" in out
    assert "sha256" in out


def test_projects_export_of_an_unknown_project_is_not_found(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """An export must never report success for a project that is not there."""
    _project_sessions(tmp_path, monkeypatch)
    code = main(["projects", "export", "ghost"])
    out, combined = _captured(capsys)
    assert code == 4
    assert "nothing to export" in combined
    assert "NOT_EXECUTED" not in out
    assert _archives(tmp_path / "exports") == []


def test_projects_export_over_the_threshold_needs_confirmation(tmp_path,
                                                               monkeypatch,
                                                               capsys):
    _project_sessions(tmp_path, monkeypatch)
    monkeypatch.setenv("MECH_EXPORT_CONFIRM_BYTES", "1")
    code = main(["projects", "export", "alpha"])
    out, combined = _captured(capsys)
    assert code == 2
    assert "estimated" in combined.lower()
    assert "mech projects export alpha --yes" in combined
    assert _archives(tmp_path / "exports") == []


def test_projects_export_over_the_threshold_with_yes(tmp_path, monkeypatch,
                                                     capsys):
    _project_sessions(tmp_path, monkeypatch)
    monkeypatch.setenv("MECH_EXPORT_CONFIRM_BYTES", "1")
    code = main(["projects", "export", "alpha", "--yes"])
    out, _ = _captured(capsys)
    assert code == 0
    assert (tmp_path / "exports" / "alpha.interp-project").is_file()


# ---------------------------------------------------------------------------
# SAE
# ---------------------------------------------------------------------------


def test_sae_list_reports_the_reference_registry(capsys):
    assert main(["sae", "list"]) == 0
    out, _ = _captured(capsys)
    assert "hook_resid_pre" in out
    assert "reference" in out
    for layer in ("8", "10"):
        assert layer in out


def test_sae_inspect_without_a_local_checkpoint_is_not_executed(
    monkeypatch, capsys
):
    monkeypatch.delenv("MECH_SAE_SOURCE", raising=False)
    monkeypatch.delenv("MECH_SAE_CHECKPOINT", raising=False)

    code = main(["sae", "inspect", "8"])
    out, combined = _captured(capsys)

    assert code == 3
    assert "NOT_EXECUTED" in out
    assert "no SAE checkpoint is configured" in combined
    assert "feature" in combined  # names what was NOT produced


def test_sae_inspect_unknown_id_is_not_found(capsys):
    code = main(["sae", "inspect", "999"])
    _, combined = _captured(capsys)

    assert code == 4
    assert "unknown SAE id" in combined


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_status_is_read_only(capsys):
    assert main(["cache", "status", "--json"]) == 0
    out, _ = _captured(capsys)
    payload = json.loads(out)

    assert payload["read_only"] is True
    assert "no cache was modified or deleted" in payload["note"]
    assert any(row["cache"] == "storage database" for row in payload["caches"])
