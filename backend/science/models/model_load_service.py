"""The control-plane seam for small modern LLMs, or an honest refusal.

Why this module exists
----------------------
`small_models.py` holds a working loader, a registry whose shapes were read from
each checkpoint's own ``config.json``, and a refusal type that carries its reason.
Nothing outside that file referenced any of it, so the capability was present and
unreachable: the model-load endpoint could only ever answer for GPT-2.

This is the seam. It answers one question -- *can this name be served by a real
small model here, and if so with which measured numbers?* -- and it answers it
without ever substituting something that looks like an answer.

The three functions
-------------------
``is_small_model(name)``
    A predicate about the registry, aliases included. ``False`` is a routing
    signal: the caller falls through to the GPT-2 engine path. A false positive
    would silently reroute ``gpt2-small`` -- the default in the load payload --
    away from the weights it is supposed to use, which is why this is a registry
    lookup and not a substring match.

``catalog_entries()``
    One row per registry entry, each labelled ``provenance: "reference"``. A
    catalogue is a list of claims about checkpoints; *nothing in it executed a
    model*, so ``live`` would be a claim about a measurement that never
    happened. What separates a reference-only name from a loadable one is
    ``load_state`` (``local-weights-cached`` / ``not-downloaded`` / ``gated``)
    and ``loadable_now``: both say something about the local cache, neither
    claims a load happened. ``loadable_now`` is not evidence of anything except
    that the bytes are on this disk.

``load_model_request(payload)``
    Loads a registered small model and returns its **measured** numbers, or
    returns an ``unavailable`` record carrying the loader's own reason verbatim.
    An unregistered name raises ``KeyError`` so the caller can fall through.

What ``live`` means here, and what cannot produce it
----------------------------------------------------
``provenance: "live"`` on the success path is issued by
``backend.core.provenance.attest_measurement`` -- the repository's single issuer
of both that label and ``attested``. This module never writes ``"live"`` itself,
which is what makes the label mean "this measurement boundary completed a load"
rather than "this code path was reachable". If the loader raises, there is no
load, so there is no ``live``: the refusal path cannot reach the attestation
call at all.

Measured means measured
-----------------------
Every number in a successful response is read off the weights that were loaded:

* the seven architecture fields come from the loaded model's own ``config`` --
  not from the registry table. Where the table and the config disagree the config
  wins, and ``shape_disagreements`` carries the difference rather than dropping
  it. A field the config does not declare is reported as ``None`` and named in
  ``not_declared_by_config``; a registry value is never substituted for a missing
  measurement.
* ``n_parameters`` and ``weight_bytes`` are summed from the live parameter
  tensors, so they describe the bytes actually held.
* ``arch`` is the class of the object that was constructed, and
  ``arch_matches_registry`` records whether it is the architecture the table
  claims -- a mismatch is visible, not averaged away.

The registry's own claims -- ``params_millions``, ``approx_fp16_vram_gb``,
``licence``, and the shape columns -- are deliberately **absent** from a load
response. They are published metadata derived from Hub file sizes, they were not
measured by this call, and a reader quoting them beside real dimensions would
have no way to tell which was which. They live in ``catalog_entries()``, marked
``reference``.

Identity, not the requested spelling
------------------------------------
``model_name`` is the registry id whose weights were loaded; the caller's raw
string is echoed as ``model_name_as_requested``. An alias
(``qwen2.5-0.5b-instruct``) resolves to the same checkpoint by construction, so
``model_requested`` is reported as the *loaded* id and a documented synonym is
not recorded as ``model_mismatch`` -- a false mismatch reading "you got the wrong
model" is worse than no mismatch flag. ``alias_resolved`` records that a synonym
was used, so the transformation stays visible either way. The two fields that echo
caller input are left unmarked in ``field_provenance`` on purpose: they are
inputs, not measurements, and marking them ``live`` would be the same error as
calling a name a fingerprint.

Refusals are not failures, and never placeholders
-------------------------------------------------
An unavailable response carries ``status: "unavailable"``,
``provenance: "unavailable"``, ``attested: False``, the loader's reason verbatim,
and no architecture or weight field at all. There is no placeholder model, no
zeroed dimension, and no score. ``NOT_EXECUTED`` is reported as exactly that.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.core.provenance import attest_measurement

from .small_models import (
    SmallModelSpec,
    SmallModelUnavailable,
    available_small_models,
    load_small_model,
    resolve,
)

LIVE = "live"
REFERENCE = "reference"
UNAVAILABLE = "unavailable"

#: Default model name, matching the dispatcher's own default. It is not a small
#: model, so `load_model_request({})` raises KeyError and the caller keeps the
#: GPT-2 path it had before.
DEFAULT_MODEL_NAME = "gpt2-small"

#: Catalogue rows describe published checkpoints; they are never measurements.
CATALOG_PROVENANCE = REFERENCE

#: Load states a catalogue row can report. Only the first means the weights are
#: on this disk; none of them means a forward pass ran.
LOAD_STATE_CACHED = "local-weights-cached"
LOAD_STATE_ABSENT = "not-downloaded"
LOAD_STATE_GATED = "gated"

#: Keys in a load response whose value the registry table also claims. A
#: disagreement is reported under the config's own name, so comparing the table
#: against the checkpoint does not need a second request.
_SHAPE_FIELDS = (
    ("n_layers", "num_hidden_layers"),
    ("n_heads", "num_attention_heads"),
    ("n_kv_heads", "num_key_value_heads"),
    ("d_model", "hidden_size"),
    ("d_mlp", "intermediate_size"),
    ("vocab_size", "vocab_size"),
    ("context_length", "max_position_embeddings"),
)


def is_small_model(name: Any) -> bool:
    """Whether ``name`` is a registered small model, aliases included.

    ``False`` means "not mine": the caller must fall through to whatever served
    this name before, rather than being handed a substitute.
    """
    try:
        resolve(name)  # type: ignore[arg-type]
    except (KeyError, TypeError, AttributeError, ValueError):
        return False
    return True


def _load_state(row: Dict[str, Any]) -> str:
    """Where this entry stands with respect to the local cache.

    ``status`` distinguishes a gated repo (refused on purpose, no request made)
    from one that is merely absent, and that distinction is the whole reason the
    gate is reported rather than attempted.
    """
    if "gated" in str(row.get("status", "")).lower():
        return LOAD_STATE_GATED
    if row.get("locally_cached"):
        return LOAD_STATE_CACHED
    return LOAD_STATE_ABSENT


def catalog_entries() -> List[Dict[str, Any]]:
    """One honest row per registry entry, for a UI or a status report.

    Every row is ``provenance: "reference"`` and every field it reports is
    labelled ``reference``: these are published facts about checkpoints, and no
    model was executed to produce them. ``load_state`` and ``loadable_now``
    separate a name that is only a reference from one whose weights are present
    locally -- without either of them being a claim that a load succeeded.
    """
    rows: List[Dict[str, Any]] = []
    for source in available_small_models():
        row = dict(source)
        row["provenance"] = CATALOG_PROVENANCE
        row["load_state"] = _load_state(source)
        row["field_provenance"] = {
            key: CATALOG_PROVENANCE
            for key in row
            if key not in ("provenance", "field_provenance")
        }
        rows.append(row)
    return rows


def _resolve(name: str) -> SmallModelSpec:
    return resolve(name)


def _declared_shape(config: Any) -> Dict[str, Optional[int]]:
    """The architecture the loaded checkpoint declares, or ``None`` per field.

    ``None`` is kept rather than filled in from the registry, because "the config
    did not say" and "the config said 30" are different facts and only one of
    them is a measurement.
    """
    shape: Dict[str, Optional[int]] = {}
    for field, attribute in _SHAPE_FIELDS:
        value = getattr(config, attribute, None)
        shape[field] = int(value) if isinstance(value, (int, float)) else None
    return shape


def _measure_weights(model: Any) -> Dict[str, int]:
    """Parameter count and held bytes, summed from the live tensors."""
    parameters = list(model.parameters()) if model is not None else []
    return {
        "n_parameters": sum(int(p.numel()) for p in parameters),
        "weight_bytes": sum(int(p.numel()) * int(p.element_size())
                            for p in parameters),
    }


def _unavailable(name: str, entry: Optional[SmallModelSpec], reason: str,
                 *, gated: bool) -> Dict[str, Any]:
    """A refusal: the reason, the echoed identifiers, and nothing measured.

    No architecture field appears, not even a zero: an echoed name labelled
    ``unavailable`` is the only thing here a caller could mistake for data, and
    the field map says so.
    """
    result: Dict[str, Any] = {
        "status": UNAVAILABLE,
        "provenance": UNAVAILABLE,
        "attested": False,
        "model_name": name,
        "small_model": True,
        "gated": bool(gated),
        "reason": reason,
    }
    if entry is not None:
        result["hf_repo_id"] = entry.hf_repo_id
    result["field_provenance"] = {
        key: UNAVAILABLE for key in result
        if key != "field_provenance"
    }
    return result


def load_model_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Load a registered small model, or return the reason it is not available.

    ``payload`` fields read here: ``model_name`` (default ``"gpt2-small"``) and
    ``allow_download`` (default ``False``). A name outside the registry raises
    ``KeyError(name)`` so a caller can fall through to its existing model path.

    The success record's ``provenance`` and ``attested`` come from
    ``attest_measurement`` -- the repository's single issuer of both. This
    function never writes ``"live"``; the label appears only if the loader
    returned a loaded adapter.
    """
    payload = payload if isinstance(payload, dict) else {}
    name = str(payload.get("model_name", DEFAULT_MODEL_NAME))
    entry = _resolve(name)

    allow_download = bool(payload.get("allow_download", False))
    try:
        adapter = load_small_model(name, allow_download=allow_download)
    except SmallModelUnavailable as exc:
        # NOT_EXECUTED, distinct from FAILED. The loader's reason is returned
        # verbatim -- rewriting it here would drop the repository name or the
        # switch that a reader needs to act on it.
        return _unavailable(name, entry, str(exc), gated=bool(exc.gated))

    model = getattr(adapter, "_model", None)
    if model is None:  # pragma: no cover - the loader never returns this
        return _unavailable(
            name, entry,
            f"{entry.hf_repo_id} was reported as loaded but carries no model "
            f"object, so nothing was measured.",
            gated=False)

    shape = _declared_shape(getattr(model, "config", None))
    weights = _measure_weights(model)
    architecture = type(model).__name__

    result: Dict[str, Any] = {
        "status": "loaded",
        "model_name": entry.model_id,
        # The caller's own string, echoed unmarked: it is an input.
        "model_name_as_requested": name,
        "alias_resolved": name.strip().lower() != entry.model_id,
        "small_model": True,
        "arch": architecture,
        "arch_matches_registry": architecture == entry.arch,
        "hf_repo_id": entry.hf_repo_id,
        **shape,
        **weights,
        "device": getattr(adapter, "device", None),
        "precision": getattr(adapter, "precision", None),
        "downloaded": bool(getattr(adapter, "downloaded", False)),
        "shape_disagreements": list(
            getattr(adapter, "shape_disagreements", []) or []),
        "not_declared_by_config": sorted(
            field for field, value in shape.items() if value is None),
        "reason": None,
    }

    # `fields` is explicit rather than inferred. `attest_measurement` labels
    # every key it is not told about as `live`, which would stamp the registry
    # echo above as a measurement; naming the measured keys keeps the map a
    # statement about this load.
    measured_fields = [
        "model_name", "arch", "hf_repo_id",
        *(field for field, _ in _SHAPE_FIELDS),
        "n_parameters", "weight_bytes", "device", "precision", "downloaded",
        "shape_disagreements",
    ]
    result = attest_measurement(
        result,
        model_loaded=entry.model_id,
        # The *loaded* id, not the requested spelling: an alias is the same
        # checkpoint, so reporting it as a mismatch would be a false alarm. The
        # requested string is echoed under `model_name_as_requested`.
        model_requested=entry.model_id,
        fields=measured_fields,
    )
    if "field_provenance" not in result:  # pragma: no cover - attestation refused
        return _unavailable(
            name, entry,
            "the attestation boundary withheld this load, so it cannot be "
            "reported as a measurement.",
            gated=False)
    return result


__all__ = [
    "LIVE",
    "REFERENCE",
    "UNAVAILABLE",
    "DEFAULT_MODEL_NAME",
    "LOAD_STATE_CACHED",
    "LOAD_STATE_ABSENT",
    "LOAD_STATE_GATED",
    "is_small_model",
    "catalog_entries",
    "load_model_request",
]
