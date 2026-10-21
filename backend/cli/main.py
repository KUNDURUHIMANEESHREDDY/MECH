"""`mech` - command line access to MECH's experiment engine (spec §22).

Thin by contract: each subcommand calls the same backend service the desktop UI
calls and prints what that service returned.

    doctor              environment and capability report
    models list         the model catalogue the UI shows
    models inspect ID   architecture of the weights actually in memory
    run forward         live next-token distribution (`/gpt2/run_prompt`)
    run logit-lens      per-layer logit lens (`/gpt2/logit_lens_all`)
    experiments list    DesktopStorage experiments
    experiments show ID one stored experiment
    experiments export ID --format markdown
    projects export ID   one project's sessions as an .interp-project archive
    sae list            known SAE checkpoints (reference metadata)
    sae inspect ID      registry facts + configured-checkpoint state
    cache status        read-only sizes of the local caches

Reuse, not reimplementation: the command handlers below call
``backend.api.dispatcher`` routes (the UI's own handlers),
``backend.services.gpt2_engine``, ``backend.interpretability.sae`` and
``backend.storage.DesktopStorage``. No inference, schema or metric is
recomputed here.

Scientific integrity (AGENTS.md): when the engine, the weights or an SAE
checkpoint is unavailable, the command exits non-zero and prints
``NOT_EXECUTED`` with the concrete reason. It never prints a fabricated table,
a zero, or a placeholder metric. It never starts a download itself: the engine
prefers the local Hugging Face cache and only falls back to the hub when the
snapshot is missing, which ``doctor`` reports rather than denies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Exit codes. Any non-zero code means "no result was produced".
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_EXECUTED = 3
EXIT_NOT_FOUND = 4

#: Mirrors ``backend.api.dispatcher._MAX_PROMPT_CHARS``. The route validates
#: again; this copy only fails fast, before any model is touched.
MAX_PROMPT_CHARS = 2048

#: Statuses that mean "the service produced a result". Everything else is a
#: refusal and is reported as NOT_EXECUTED.
_OK_STATUSES = frozenset({"ok", "loaded", "live", "completed"})

#: Bounded directory scan for `cache status`; never follows links.
_MAX_CACHE_ENTRIES = 20_000


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Aligned plain-text table. Values are printed as-is, never reformatted."""
    cells = [["" if value is None else str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in cells:
        for index, value in enumerate(row[: len(widths)]):
            widths[index] = max(widths[index], len(value))
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip())
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _not_executed(
    reason: str,
    *,
    as_json: bool = False,
    engine: Optional[str] = None,
    capability: Optional[str] = None,
) -> int:
    """Report a refusal. Always non-zero, always labelled NOT_EXECUTED."""
    from backend.core.execution_status import not_executed_payload

    payload = not_executed_payload(reason, engine=engine, capability=capability)
    if as_json:
        _print_json(payload)
    else:
        print("NOT_EXECUTED:", reason)
    return EXIT_NOT_EXECUTED


def _status(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("status") or "").strip().lower()
    return ""


def _is_ok(payload: Any) -> bool:
    return _status(payload) in _OK_STATUSES


def _reason(payload: Any, fallback: str = "the engine returned no result") -> str:
    if isinstance(payload, dict):
        for key in ("reason", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:500]
        status = payload.get("status")
        if status:
            return f"the engine returned status={status!r}"[:500]
    return fallback


def _first(record: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if record.get(key) not in (None, ""):
            return record[key]
    return None


def _render_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return "" if value is None else str(value)


def _store() -> Tuple[Any, Path]:
    """The UI's storage authority and the path it resolved."""
    from backend.storage import DesktopStorage, get_default_db_path

    path = get_default_db_path()
    store = DesktopStorage(path)
    store.initialize()
    return store, path


def _read_prompt_file(path_value: str) -> Tuple[Optional[str], Optional[int]]:
    """Validate a prompt file before anything is executed.

    Returns ``(text, None)`` or ``(None, exit_code)``. Config files are checked
    before use: a missing, empty, non-UTF-8 or oversized file is a usage error,
    not a run with an empty prompt.
    """
    path = Path(path_value)
    if not path.exists():
        _error(f"prompt file does not exist: {path}")
        return None, EXIT_USAGE
    if not path.is_file():
        _error(f"prompt file is not a regular file: {path}")
        return None, EXIT_USAGE
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        _error(f"prompt file is not valid UTF-8 ({path}): {exc}")
        return None, EXIT_USAGE
    except OSError as exc:
        _error(f"prompt file could not be read ({path}): {exc}")
        return None, EXIT_USAGE
    text = text.strip()
    if not text:
        _error(f"prompt file is empty: {path}")
        return None, EXIT_USAGE
    if len(text) > MAX_PROMPT_CHARS:
        _error(
            f"prompt file exceeds {MAX_PROMPT_CHARS} characters "
            f"({len(text)} in {path}); the engine rejects it too"
        )
        return None, EXIT_USAGE
    return text, None


def _engine_and_model() -> Tuple[Any, Optional[str]]:
    """The GPT-2 engine module and the checkpoint id it serves, or (None, None)."""
    try:
        from backend.services import gpt2_engine
    except Exception:
        return None, None
    return gpt2_engine, getattr(gpt2_engine, "SUPPORTED_MODEL", None)


def _resolve_run_prompt(args: argparse.Namespace) -> Tuple[Optional[str], Optional[int]]:
    """Prompt from --prompt-file or --prompt, validated before execution."""
    file_value = getattr(args, "prompt_file", None)
    inline = getattr(args, "prompt", None)
    if file_value and inline:
        _error("--prompt and --prompt-file are mutually exclusive")
        return None, EXIT_USAGE
    if file_value:
        return _read_prompt_file(file_value)
    if inline:
        text = inline.strip()
        if not text:
            _error("--prompt is empty")
            return None, EXIT_USAGE
        if len(text) > MAX_PROMPT_CHARS:
            _error(f"--prompt exceeds {MAX_PROMPT_CHARS} characters")
            return None, EXIT_USAGE
        return text, None
    return None, None


def _check_model(model: str, as_json: bool) -> Optional[int]:
    """Refuse a model the engine cannot serve, before anything is loaded.

    The engine keeps one module-level model and hardcodes the checkpoint it
    loads; running "gpt2" while the caller asked for "gpt2-medium" would label
    one model's numbers with another model's name. So the request is refused.
    """
    engine, supported = _engine_and_model()
    if engine is None:
        return _not_executed(
            "torch/transformers is not importable in this interpreter, so no "
            "model can be loaded and nothing was executed.",
            as_json=as_json,
            engine="backend.services.gpt2_engine",
            capability="models",
        )
    if supported and model != supported:
        return _not_executed(
            f"this build loads only the {supported!r} checkpoint; {model!r} has "
            "no loader in backend.services.gpt2_engine, so its weights were not "
            "loaded and no result is reported for it.",
            as_json=as_json,
            engine="backend.services.gpt2_engine",
            capability="models",
        )
    return None


def _weights_cache_state() -> Tuple[bool, str]:
    """Whether the GPT-2 snapshot is in the local cache, and what a load does.

    ``gpt2_engine.load()`` tries ``local_files_only=True`` first and falls back
    to a plain ``from_pretrained("gpt2")`` -- a hub fetch -- when the snapshot
    is absent. So "weights are never downloaded" is not a true statement and
    must not be printed; the state is probed instead (tokenizer only, no
    weights, no network) and a failed probe says so rather than reporting
    either answer.
    """
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception as exc:  # absent cache, unimportable stack, bad mirror
        return False, (
            "GPT-2 weights are not in the local cache; the first `run` would "
            f"use the engine's hub fallback. Probe: {str(exc)[:120]}"
        )
    return True, "GPT-2 weights are cached locally; a load stays offline"


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks: List[Dict[str, Any]] = []

    python_ok = sys.version_info >= (3, 11)
    checks.append(
        {
            "check": "python",
            "ok": python_ok,
            "critical": True,
            "detail": f"{sys.version.split()[0]} at {sys.executable} "
                      f"(requires >= 3.11)",
        }
    )

    engine, supported = _engine_and_model()
    if engine is None:
        checks.append(
            {
                "check": "ml stack (torch/transformers)",
                "ok": False,
                "critical": False,
                "detail": "not importable; run/logit-lens will report NOT_EXECUTED",
            }
        )
    else:
        available = bool(engine.is_available())
        checks.append(
            {
                "check": "ml stack (torch/transformers)",
                "ok": available,
                "critical": False,
                "detail": (
                    "importable"
                    if available
                    else "imported but reports unavailable; nothing can run"
                ),
            }
        )
        cached, cache_detail = _weights_cache_state()
        checks.append(
            {
                "check": "engine checkpoints",
                "ok": cached,
                "critical": False,
                "detail": f"serves {supported!r} only; {cache_detail}",
            }
        )

    try:
        from backend.storage import get_default_db_path

        db_path = Path(get_default_db_path())
        parent = db_path if db_path.is_dir() else db_path.parent
        writable = parent.exists() and os.access(parent, os.W_OK)
        if db_path.exists():
            detail = f"{db_path} ({db_path.stat().st_size} bytes, writes allowed={writable})"
        else:
            detail = f"{db_path} (absent; created on first write, writes allowed={writable})"
        checks.append(
            {
                "check": "storage",
                "ok": bool(writable),
                "critical": True,
                "detail": detail,
            }
        )
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        checks.append(
            {
                "check": "storage",
                "ok": False,
                "critical": True,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )

    try:
        from backend.interpretability.sae.loader import SAELoader  # noqa: F401

        sae_ok, sae_detail = True, "loader importable"
    except Exception as exc:  # noqa: BLE001
        sae_ok, sae_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(
        {"check": "sae loader", "ok": sae_ok, "critical": False, "detail": sae_detail}
    )

    configured = bool(os.environ.get("MECH_SAE_CHECKPOINT", "").strip())
    checks.append(
        {
            "check": "sae checkpoint",
            "ok": configured,
            "critical": False,
            "detail": (
                f"configured: {os.environ.get('MECH_SAE_SOURCE', '')!r} / "
                f"{os.environ.get('MECH_SAE_CHECKPOINT', '')!r}"
                if configured
                else "none configured (MECH_SAE_SOURCE / MECH_SAE_CHECKPOINT); "
                     "sae inspect reports NOT_EXECUTED"
            ),
        }
    )

    critical_ok = all(check["ok"] for check in checks if check["critical"])
    if args.json:
        _print_json(
            {
                "status": "ok" if critical_ok else "error",
                "entrypoint": "backend.cli.main:main",
                "checks": checks,
                "offline": True,
            }
        )
    else:
        print("MECH doctor")
        _table(
            ["RESULT", "CHECK", "DETAIL"],
            [
                ["PASS" if check["ok"] else "FAIL", check["check"], check["detail"]]
                for check in checks
            ],
        )
        if not critical_ok:
            print()
            print("A critical check failed; commands that need it will report NOT_EXECUTED.")
    return EXIT_OK if critical_ok else EXIT_ERROR


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def _cmd_models_list(args: argparse.Namespace) -> int:
    try:
        from backend.api import dispatcher

        payload = dispatcher.list_models()
    except Exception as exc:  # noqa: BLE001 - refusal, not an empty table
        return _not_executed(
            f"model catalogue unavailable: {type(exc).__name__}: {exc}",
            as_json=args.json,
            engine="backend.api.dispatcher",
            capability="models",
        )

    if args.json:
        _print_json(payload)
        return EXIT_OK

    models = payload.get("models") or []
    print("Model catalogue (reference: published checkpoint names, no model executed)")
    _table(["MODEL"], [[name] for name in models])
    small = payload.get("small_models") or []
    if small:
        print()
        print("Small-model registry (published facts; load_state says whether weights are local)")
        _table(
            ["MODEL_ID", "HF_REPO_ID", "PARAMS(M)", "N_LAYERS", "D_MODEL", "LOAD_STATE"],
            [
                [
                    _first(row, ("model_id", "id", "name")),
                    row.get("hf_repo_id"),
                    row.get("params_millions"),
                    row.get("n_layers"),
                    row.get("d_model"),
                    _render_value(row.get("load_state")),
                ]
                for row in small
            ],
        )
    if payload.get("small_models_error"):
        print()
        print(f"warning: small-model registry error: {payload['small_models_error']}")
    print()
    print("provenance: reference - no weights were loaded by this command.")
    return EXIT_OK


def _cmd_models_inspect(args: argparse.Namespace) -> int:
    name: str = args.model_id
    engine, supported = _engine_and_model()
    if engine is None:
        return _not_executed(
            "torch/transformers is not importable in this interpreter, so no "
            "architecture can be read from real weights.",
            as_json=args.json,
            engine="backend.services.gpt2_engine",
            capability="models",
        )
    if supported and name != supported:
        return _not_executed(
            f"this build loads only the {supported!r} checkpoint; {name!r} has no "
            "loader, so its architecture was not read from any weights.",
            as_json=args.json,
            engine="backend.services.gpt2_engine",
            capability="models",
        )

    try:
        from backend.api import dispatcher

        loaded = dispatcher.gpt2_load({})
        if not _is_ok(loaded):
            return _not_executed(
                f"the {name!r} checkpoint could not be loaded: {_reason(loaded)}",
                as_json=args.json,
                engine="backend.services.gpt2_engine",
                capability="models",
            )
        info = dispatcher.get_model_info(name)
    except Exception as exc:  # noqa: BLE001
        return _not_executed(
            f"model inspection failed: {type(exc).__name__}: {exc}",
            as_json=args.json,
            engine="backend.api.dispatcher",
            capability="models",
        )

    if info.get("model_mismatch") or not _is_ok(info):
        return _not_executed(
            _reason(info, "the loaded weights do not match the requested model"),
            as_json=args.json,
            engine="backend.services.gpt2_engine",
            capability="models",
        )

    if args.json:
        _print_json(info)
        return EXIT_OK

    print(f"Model: {info.get('model_name')}")
    fields = [
        ("model_type", info.get("model_type")),
        ("architecture", info.get("architecture")),
        ("n_layers", info.get("n_layers")),
        ("n_heads", info.get("n_heads")),
        ("d_model", info.get("d_model")),
        ("d_mlp", info.get("d_mlp")),
        ("d_head", info.get("d_head")),
        ("vocab_size", info.get("vocab_size")),
        ("n_positions", info.get("n_positions")),
        ("n_params", info.get("n_params")),
        ("device", info.get("device")),
        ("dtype", info.get("dtype")),
        ("activation", info.get("activation")),
        ("provenance", info.get("provenance")),
        ("attested", info.get("attested")),
    ]
    _table(["FIELD", "VALUE"], [[key, _render_value(value)] for key, value in fields])
    return EXIT_OK


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _cmd_run_forward(args: argparse.Namespace) -> int:
    prompt, code = _resolve_run_prompt(args)
    if code is not None:
        return code
    refusal = _check_model(args.model, args.json)
    if refusal is not None:
        return refusal
    if prompt is None:
        _error("run forward needs --prompt or --prompt-file")
        return EXIT_USAGE

    try:
        from backend.api import dispatcher

        result = dispatcher.gpt2_run_prompt({"prompt": prompt})
    except Exception as exc:  # noqa: BLE001
        return _not_executed(
            f"forward pass failed: {type(exc).__name__}: {exc}",
            as_json=args.json,
            engine="backend.services.gpt2_engine",
            capability="forward",
        )

    if not _is_ok(result):
        return _not_executed(
            _reason(result),
            as_json=args.json,
            engine="backend.services.gpt2_engine",
            capability="forward",
        )

    if args.json:
        _print_json(result)
        return EXIT_OK

    print("Prompt:", result.get("prompt", prompt))
    tokens = result.get("str_tokens") or []
    if tokens:
        print("Tokens:", " | ".join(str(token) for token in tokens))
    top5 = result.get("top5") or []
    if top5:
        print()
        _table(
            ["TOKEN", "LOGIT", "PROB"],
            [
                [row.get("token"), row.get("logit"), row.get("prob")]
                for row in top5
                if isinstance(row, dict)
            ],
        )
    print()
    print("next_token:", result.get("next_token"))
    print(f"provenance: {result.get('provenance')} "
          f"(attested={result.get('attested')}, "
          f"model_loaded={result.get('model_loaded') or result.get('model_id')})")
    fingerprint = result.get("model_content_hash") or result.get("weights_fingerprint")
    if fingerprint:
        print(f"weights sha256: {fingerprint}")
    else:
        print("weights sha256: unavailable (the engine did not attest one)")
    return EXIT_OK


def _cmd_run_logit_lens(args: argparse.Namespace) -> int:
    prompt, code = _resolve_run_prompt(args)
    if code is not None:
        return code
    refusal = _check_model(args.model, args.json)
    if refusal is not None:
        return refusal
    if prompt is None:
        # Same default the UI route uses, named in the output.
        prompt = "The capital of France is"

    try:
        from backend.api import dispatcher

        result = dispatcher.gpt2_logit_lens_all({"prompt": prompt})
    except Exception as exc:  # noqa: BLE001
        return _not_executed(
            f"logit lens failed: {type(exc).__name__}: {exc}",
            as_json=args.json,
            engine="backend.services.gpt2_engine",
            capability="logit_lens",
        )

    if not _is_ok(result):
        return _not_executed(
            _reason(result),
            as_json=args.json,
            engine="backend.services.gpt2_engine",
            capability="logit_lens",
        )

    if args.json:
        _print_json(result)
        return EXIT_OK

    print("Prompt:", result.get("prompt", prompt))
    rows = []
    for layer in result.get("layers") or []:
        top_k = layer.get("top_k_tokens") or []
        rendered = ", ".join(
            str(item.get("token")) if isinstance(item, dict) else str(item)
            for item in top_k
        )
        rows.append([layer.get("layer"), layer.get("top_token"), rendered])
    _table(["LAYER", "TOP_TOKEN", "TOP_K_TOKENS"], rows)
    print()
    print("caveat: the logit lens projects each layer's residual stream through "
          "the output head; it is a projection of an intermediate state, not the "
          "model's own per-layer prediction.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


def _experiments(args: argparse.Namespace) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Path], Optional[int]]:
    try:
        store, path = _store()
        items = store.list_experiments()
    except Exception as exc:  # noqa: BLE001 - storage failure, not an empty list
        _error(f"storage unavailable: {type(exc).__name__}: {exc}")
        return None, None, EXIT_ERROR
    return items, path, None


def _find_experiment(
    items: Sequence[Dict[str, Any]], experiment_id: str
) -> Optional[Dict[str, Any]]:
    for item in items:
        if _first(item, ("id", "experiment_id", "run_id")) == experiment_id:
            return item
    return None


def _cmd_experiments_list(args: argparse.Namespace) -> int:
    items, path, code = _experiments(args)
    if code is not None:
        return code
    assert items is not None and path is not None

    if args.json:
        _print_json(
            {
                "status": "ok",
                "storage": str(path),
                "count": len(items),
                "experiments": items,
                "provenance": "live" if items else "none",
            }
        )
        return EXIT_OK

    if not items:
        print(f"No experiments recorded in {path}.")
        return EXIT_OK
    print(f"Experiments in {path} ({len(items)})")
    _table(
        ["ID", "NAME", "STATUS", "CREATED_AT"],
        [
            [
                _first(item, ("id", "experiment_id", "run_id")),
                _first(item, ("name", "title")),
                item.get("status"),
                item.get("created_at"),
            ]
            for item in items
        ],
    )
    return EXIT_OK


def _cmd_experiments_show(args: argparse.Namespace) -> int:
    items, path, code = _experiments(args)
    if code is not None:
        return code
    assert items is not None and path is not None

    record = _find_experiment(items, args.experiment_id)
    if record is None:
        _error(f"experiment {args.experiment_id!r} not found in {path}")
        return EXIT_NOT_FOUND

    if args.json:
        _print_json(record)
        return EXIT_OK

    print(f"Experiment {args.experiment_id} (stored in {path})")
    _table(
        ["FIELD", "VALUE"],
        [[key, _render_value(record[key])] for key in sorted(record)],
    )
    return EXIT_OK


def _markdown_export(record: Dict[str, Any], path: Path) -> str:
    experiment_id = _first(record, ("id", "experiment_id", "run_id"))
    lines = [
        f"# Experiment `{experiment_id}`",
        "",
        f"Exported from `{path}` by `mech experiments export`. Fields are exactly ",
        "as stored; nothing was computed, rounded or added by the exporter.",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key in sorted(record):
        value = _render_value(record[key]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{key}` | {value} |")
    lines += [
        "",
        "## Raw record",
        "",
        "```json",
        json.dumps(record, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


def _cmd_experiments_export(args: argparse.Namespace) -> int:
    items, path, code = _experiments(args)
    if code is not None:
        return code
    assert items is not None and path is not None

    record = _find_experiment(items, args.experiment_id)
    if record is None:
        _error(f"experiment {args.experiment_id!r} not found in {path}")
        return EXIT_NOT_FOUND

    if args.format == "json":
        _print_json(record)
    else:
        print(_markdown_export(record, path))
    return EXIT_OK


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------


def _cmd_projects_export(args: argparse.Namespace) -> int:
    """Export one project's sessions as a .interp-project archive.

    The archive goes to the server-side export root (``MECH_EXPORT_ROOT``,
    else ``~/.mech/exports``). This command has no destination flag: the
    destination is not the caller's to choose.
    """
    from backend.api.project_export import (
        InvalidProjectId,
        ProjectExporter,
        ProjectNotFound,
        estimate_export,
    )

    project_id = args.project_id
    try:
        estimate = estimate_export(project_id)
    except InvalidProjectId as exc:
        _error(str(exc))
        return EXIT_USAGE
    except ValueError as exc:
        _error(f"the export could not be estimated: {exc}")
        return EXIT_ERROR

    # A project no stored session names does not exist. Reporting success
    # for it would be a lie: there would be nothing in the archive.
    if estimate["sessions_included"] == 0:
        _error(f"no stored session names project {project_id!r}; "
               f"there is nothing to export")
        return EXIT_NOT_FOUND

    if args.estimate:
        print(f"Export estimate for project {project_id} (nothing was written)")
        _table(
            ["FIELD", "VALUE"],
            [
                ["sessions_included", estimate["sessions_included"]],
                ["sessions_omitted_unattributed",
                 estimate["sessions_omitted_unattributed"]],
                ["sessions_omitted_other_projects",
                 estimate["sessions_omitted_other_projects"]],
                ["estimated_bytes", estimate["estimated_bytes"]],
                ["threshold_bytes", estimate["threshold_bytes"]],
                ["requires_confirmation", estimate["requires_confirmation"]],
                ["estimate_basis", estimate["estimate_basis"]],
            ],
        )
        return EXIT_OK

    if estimate["requires_confirmation"] and not args.yes:
        print(f"the export of {project_id} is estimated at "
              f"{estimate['estimated_bytes']} bytes, above the "
              f"{estimate['threshold_bytes']}-byte threshold; "
              f"nothing was written")
        print(f"re-run with: mech projects export {project_id} --yes")
        return EXIT_USAGE

    try:
        result = ProjectExporter().export_project(project_id)
    except ProjectNotFound as exc:
        _error(str(exc))
        return EXIT_NOT_FOUND
    except InvalidProjectId as exc:
        _error(str(exc))
        return EXIT_USAGE
    except ValueError as exc:
        _error(str(exc))
        return EXIT_USAGE
    except OSError as exc:
        _error(f"the archive could not be written: {exc}")
        return EXIT_ERROR

    print(f"Exported {result['archive_name']}")
    _table(
        ["FIELD", "VALUE"],
        [
            ["archive_name", result["archive_name"]],
            ["bytes_written", result["bytes_written"]],
            ["sha256", result["sha256"]],
            ["sessions_included", result["sessions_included"]],
            ["exported_at", result["exported_at"]],
        ],
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# sae
# ---------------------------------------------------------------------------


def _sae_registry() -> Any:
    from backend.interpretability.sae import registry

    return registry


def _cmd_sae_list(args: argparse.Namespace) -> int:
    try:
        registry = _sae_registry()
        entries = [registry.KNOWN_SAES[layer] for layer in registry.available_layers()]
    except Exception as exc:  # noqa: BLE001
        return _not_executed(
            f"SAE registry unavailable: {type(exc).__name__}: {exc}",
            as_json=args.json,
            engine="backend.interpretability.sae.registry",
            capability="sae",
        )

    if args.json:
        _print_json(
            {
                "status": "ok",
                "provenance": "reference",
                "saes": [entry.as_dict() for entry in entries],
            }
        )
        return EXIT_OK

    print("Known SAE checkpoints (reference metadata; no checkpoint was loaded)")
    _table(
        ["LAYER", "HOOK", "D_IN", "D_SAE", "MODEL", "DATASET", "L1", "REVISION"],
        [
            [
                entry.layer,
                entry.hook_point,
                entry.d_in,
                entry.d_sae,
                entry.model_id,
                entry.dataset,
                entry.l1_coefficient,
                entry.revision[:12],
            ]
            for entry in entries
        ],
    )
    print()
    print(f"repo: {getattr(registry, 'REPO_ID', '?')}")
    print("no checkpoint is downloaded by this command; `sae inspect` reports "
          "NOT_EXECUTED unless a local checkpoint is configured.")
    return EXIT_OK


def _parse_sae_id(value: str) -> Optional[int]:
    match = re.fullmatch(r"(?:layer)?\s*(\d+)", (value or "").strip(), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _local_checkpoint_dir() -> Path:
    configured = os.environ.get("MECH_SAE_DIR", "").strip()
    if configured:
        return Path(configured)
    return _repo_root() / "backend" / "storage" / "sae_checkpoints"


def _cmd_sae_inspect(args: argparse.Namespace) -> int:
    try:
        registry = _sae_registry()
        entries = [registry.KNOWN_SAES[layer] for layer in registry.available_layers()]
    except Exception as exc:  # noqa: BLE001
        return _not_executed(
            f"SAE registry unavailable: {type(exc).__name__}: {exc}",
            as_json=args.json,
            engine="backend.interpretability.sae.registry",
            capability="sae",
        )

    layer = _parse_sae_id(args.sae_id)
    entry = registry.resolve(layer) if layer is not None else None
    if entry is None:
        known = ", ".join(str(item.layer) for item in entries)
        _error(f"unknown SAE id {args.sae_id!r}; known layer ids: {known}")
        return EXIT_NOT_FOUND

    source = os.environ.get("MECH_SAE_SOURCE", "").strip().lower()
    identifier = os.environ.get("MECH_SAE_CHECKPOINT", "").strip()
    checkpoint_dir = _local_checkpoint_dir()
    checkpoint_path = checkpoint_dir / identifier if identifier else None
    present = bool(
        checkpoint_path is not None
        and os.path.basename(identifier) == identifier
        and checkpoint_path.is_file()
    )

    if not args.json:
        print("SAE registry entry (reference metadata, verified checkpoint facts)")
        _table(
            ["FIELD", "VALUE"],
            [
                ["layer", entry.layer],
                ["hook_point", entry.hook_point],
                ["d_in", entry.d_in],
                ["d_sae", entry.d_sae],
                ["model_id", entry.model_id],
                ["dataset", entry.dataset],
                ["l1_coefficient", entry.l1_coefficient],
                ["repo_id", entry.repo_id],
                ["revision", entry.revision],
                ["checkpoint_dir", checkpoint_dir],
                ["checkpoint_configured", bool(source and identifier)],
                ["checkpoint_present_locally", present],
            ],
        )
        print()

    if not source or not identifier:
        return _not_executed(
            f"no SAE checkpoint is configured for layer {entry.layer}, so no "
            "encoder weights were loaded and no feature decomposition is "
            "reported. Set MECH_SAE_SOURCE=local and MECH_SAE_CHECKPOINT to a "
            "filename inside the checkpoint directory to inspect it.",
            as_json=args.json,
            engine="backend.interpretability.sae.loader",
            capability="sae",
        )
    if source not in ("local", "file"):
        return _not_executed(
            f"the configured SAE source {source!r} would download weights from a "
            "remote hub; this CLI never downloads. Use a local checkpoint.",
            as_json=args.json,
            engine="backend.interpretability.sae.loader",
            capability="sae",
        )
    if not present:
        return _not_executed(
            f"checkpoint {identifier!r} is not present in {checkpoint_dir}; "
            "nothing was loaded and no download was attempted.",
            as_json=args.json,
            engine="backend.interpretability.sae.loader",
            capability="sae",
        )

    try:
        from backend.api import dispatcher

        result = dispatcher.sae_inspect(
            {
                "layer": entry.layer,
                "prompt": args.prompt,
                "source": "local",
                "identifier": identifier,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _not_executed(
            f"SAE inspection failed: {type(exc).__name__}: {exc}",
            as_json=args.json,
            engine="backend.interpretability.sae.loader",
            capability="sae",
        )

    if not _is_ok(result):
        return _not_executed(
            _reason(result),
            as_json=args.json,
            engine="backend.interpretability.sae.loader",
            capability="sae",
        )

    if args.json:
        _print_json(result)
        return EXIT_OK

    print("SAE inspection (live encoding of one hidden state)")
    _table(
        ["FIELD", "VALUE"],
        [[key, _render_value(result[key])] for key in sorted(result)
         if key != "field_provenance"],
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def _dir_metrics(path: Path) -> Dict[str, Any]:
    """Bounded, read-only size walk. Never modifies, never follows links."""
    if not path.exists():
        return {"path": str(path), "exists": False, "files": 0, "bytes": 0,
                "truncated": False}
    if path.is_file():
        return {"path": str(path), "exists": True, "files": 1,
                "bytes": path.stat().st_size, "truncated": False}

    files = 0
    total = 0
    truncated = False
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if files >= _MAX_CACHE_ENTRIES:
                truncated = True
                break
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    files += 1
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
        if truncated:
            break
    return {"path": str(path), "exists": True, "files": files,
            "bytes": total, "truncated": truncated}


def _hf_cache_dir() -> Path:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache)
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cmd_cache_status(args: argparse.Namespace) -> int:
    from backend.storage import get_default_db_path

    targets = [
        ("storage database", Path(get_default_db_path())),
        ("SAE checkpoints", _local_checkpoint_dir()),
        ("SAE activation cache", _repo_root() / "backend" / "storage" / "activation_cache"),
        ("HuggingFace cache", _hf_cache_dir()),
        ("torch cache", Path.home() / ".cache" / "torch"),
    ]
    rows = [{"cache": name, **_dir_metrics(path)} for name, path in targets]
    payload = {
        "status": "ok",
        "read_only": True,
        "note": "sizes only; no cache was modified or deleted by this command",
        "caches": rows,
    }
    if args.json:
        _print_json(payload)
        return EXIT_OK

    print("Local caches (read-only; nothing is modified or deleted)")
    _table(
        ["CACHE", "EXISTS", "FILES", "BYTES", "PATH"],
        [
            [
                row["cache"],
                "yes" if row["exists"] else "no",
                row["files"] if row["exists"] else "-",
                row["bytes"] if row["exists"] else "-",
                row["path"],
            ]
            for row in rows
        ],
    )
    if any(row["truncated"] for row in rows):
        print()
        print(f"note: at least one cache exceeded {_MAX_CACHE_ENTRIES} entries; "
              "counts are a lower bound.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true",
        help="print the backend payload as JSON instead of a table",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mech",
        description="MECH command line: the same engine and schemas as the UI.",
    )
    commands = parser.add_subparsers(dest="command", metavar="<command>")
    commands.required = True

    doctor = commands.add_parser(
        "doctor", help="report environment, storage and capability state (offline)"
    )
    _add_json_flag(doctor)
    doctor.set_defaults(func=_cmd_doctor)

    models = commands.add_parser("models", help="the model catalogue and loaded architecture")
    model_commands = models.add_subparsers(dest="models_command", metavar="<command>")
    model_commands.required = True
    models_list = model_commands.add_parser("list", help="model catalogue the UI shows")
    _add_json_flag(models_list)
    models_list.set_defaults(func=_cmd_models_list)
    models_inspect = model_commands.add_parser(
        "inspect", help="architecture read from the loaded weights"
    )
    models_inspect.add_argument("model_id", help="checkpoint id, e.g. gpt2")
    _add_json_flag(models_inspect)
    models_inspect.set_defaults(func=_cmd_models_inspect)

    run = commands.add_parser("run", help="execute the live engine")
    run_commands = run.add_subparsers(dest="run_command", metavar="<command>")
    run_commands.required = True

    def add_run_common(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--model", default="gpt2",
            help="checkpoint id; this build serves only 'gpt2'",
        )
        target.add_argument("--prompt", help="prompt text")
        target.add_argument(
            "--prompt-file", help="read the prompt from this UTF-8 file (validated first)"
        )
        _add_json_flag(target)

    forward = run_commands.add_parser("forward", help="next-token distribution")
    add_run_common(forward)
    forward.set_defaults(func=_cmd_run_forward)

    logit_lens = run_commands.add_parser("logit-lens", help="per-layer logit lens")
    add_run_common(logit_lens)
    logit_lens.set_defaults(func=_cmd_run_logit_lens)

    experiments = commands.add_parser(
        "experiments", help="stored experiments (DesktopStorage)"
    )
    experiment_commands = experiments.add_subparsers(
        dest="experiments_command", metavar="<command>"
    )
    experiment_commands.required = True
    experiments_list = experiment_commands.add_parser("list", help="list experiments")
    _add_json_flag(experiments_list)
    experiments_list.set_defaults(func=_cmd_experiments_list)
    experiments_show = experiment_commands.add_parser("show", help="one experiment")
    experiments_show.add_argument("experiment_id")
    _add_json_flag(experiments_show)
    experiments_show.set_defaults(func=_cmd_experiments_show)
    experiments_export = experiment_commands.add_parser("export", help="export one experiment")
    experiments_export.add_argument("experiment_id")
    experiments_export.add_argument(
        "--format", choices=("markdown", "json"), default="markdown",
        help="export format (default: markdown)",
    )
    experiments_export.set_defaults(func=_cmd_experiments_export)

    projects = commands.add_parser(
        "projects", help="project archives (.interp-project)")
    project_commands = projects.add_subparsers(
        dest="projects_command", metavar="<command>")
    project_commands.required = True
    projects_export = project_commands.add_parser(
        "export", help="export one project's sessions as an archive")
    projects_export.add_argument("project_id")
    projects_export.add_argument(
        "--estimate", action="store_true",
        help="print the export estimate and write nothing",
    )
    projects_export.add_argument(
        "--yes", action="store_true",
        help="confirm an export above the size threshold",
    )
    projects_export.set_defaults(func=_cmd_projects_export)

    sae = commands.add_parser("sae", help="SAE checkpoints and inspection")
    sae_commands = sae.add_subparsers(dest="sae_command", metavar="<command>")
    sae_commands.required = True
    sae_list = sae_commands.add_parser("list", help="known SAE checkpoints")
    _add_json_flag(sae_list)
    sae_list.set_defaults(func=_cmd_sae_list)
    sae_inspect = sae_commands.add_parser("inspect", help="inspect one known SAE (layer id)")
    sae_inspect.add_argument("sae_id", help="layer id, e.g. 8 or layer8")
    sae_inspect.add_argument(
        "--prompt", default="The capital of France is",
        help="prompt for the hidden state (used only when a local checkpoint exists)",
    )
    _add_json_flag(sae_inspect)
    sae_inspect.set_defaults(func=_cmd_sae_inspect)

    cache = commands.add_parser("cache", help="local cache state (read-only)")
    cache_commands = cache.add_subparsers(dest="cache_command", metavar="<command>")
    cache_commands.required = True
    cache_status = cache_commands.add_parser("status", help="sizes of local caches")
    _add_json_flag(cache_status)
    cache_status.set_defaults(func=_cmd_cache_status)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one CLI invocation. Returns a process exit code (never None)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse reports usage errors this way
        return int(exc.code or EXIT_USAGE)

    handler = getattr(args, "func", None)
    if handler is None:  # pragma: no cover - required subparsers prevent this
        parser.print_help()
        return EXIT_USAGE

    try:
        return int(handler(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130
    except Exception as exc:  # noqa: BLE001 - never turn a crash into success
        _error(f"{type(exc).__name__}: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - console-script entry point
    raise SystemExit(main())
