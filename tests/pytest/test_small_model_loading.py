"""Small modern LLMs must be reachable from the control plane, measured or refused.

What this covers
----------------
`backend/science/models/small_models.py` ships a complete, honest loader and a
registry whose shapes were read off each checkpoint's own ``config.json``. Until
now nothing outside that file referenced it: the capability existed and no caller
could reach it. `backend/science/models/model_load_service.py` is the seam, and
these are its contract tests.

The contract, and why each clause is a test
-------------------------------------------
* ``is_small_model(name)`` is a *predicate about the registry*, not a fuzzy
  matcher: a False answer is what lets the dispatcher fall through to the GPT-2
  engine path, so a false positive would silently reroute ``gpt2-small`` away
  from the weights it is supposed to use.
* A completed load returns ``provenance == "live"``, ``attested is True``, and
  architecture numbers **reconciled against the checkpoint's own config.json** --
  read here from the Hub cache by a different code path than the loader's, so the
  assertion is not the loader agreeing with itself. Weight accounting is checked
  against the safetensors file's own tensor shapes for the same reason.
* Every refusal is ``status == "unavailable"`` with the reason attached, never a
  placeholder model and never an invented architecture. ``NOT_EXECUTED`` is not
  ``FAILED``: the refusal carries no measured-looking payload.
* ``provenance == "live"`` is unobtainable without a completed load. That is
  proved by making the load fail and checking the label does not appear.
* The gated entry is refused *before any request is made*, asserted by watching
  the two `from_pretrained` entry points fail loudly if they are reached.

No test here passes because a dict came back. Each one asserts either an identity
(which weights), a computation (which numbers, against an independent read), or a
refusal (with its reason and with the absence of the thing refused).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from backend.science.models import model_load_service as svc
from backend.science.models import small_models
from backend.science.models.small_models import DOWNLOAD_ENV

#: The two registry entries the task names as locally cached. Parametrised rather
#: than hardcoded because a cache is host state: an entry that is not here skips
#: with its reason instead of failing on a machine that never fetched it.
CACHED_CANDIDATES = ("smollm2-135m", "qwen2.5-0.5b")

#: Names that must NOT be treated as small models. `gpt2-small` is the important
#: one: it is the default in the model-load payload, and a True here would take
#: the request away from the GPT-2 engine that actually holds those weights.
NON_SMALL_NAMES = ("gpt2-small", "gpt2-medium", "gemma-2b", "llama-3-8b",
                   "not-a-model", "", "   ")


# ---------------------------------------------------------------- HF cache reads
# Deliberately a second implementation of "where is this checkpoint", written
# against `huggingface_hub`'s documented layout, so the reconciliation below is
# not the loader's own path agreeing with itself.


def _snapshot_dir(repo_id: str):
    from huggingface_hub.constants import HF_HUB_CACHE

    root = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"))
    if not os.path.isdir(root):
        return None
    # `refs/main` names the revision the cache was fetched at; prefer it, because
    # a stale sibling snapshot must not be what the config is read from.
    ref = os.path.join(root, "refs", "main")
    if os.path.isfile(ref):
        try:
            revision = Path(ref).read_text(encoding="utf-8").strip()
        except OSError:
            revision = ""
        candidate = os.path.join(root, "snapshots", revision)
        if revision and os.path.isdir(candidate):
            return candidate
    snapshots = os.path.join(root, "snapshots")
    if not os.path.isdir(snapshots):
        return None
    names = sorted(os.listdir(snapshots),
                   key=lambda n: os.path.getmtime(os.path.join(snapshots, n)))
    return os.path.join(snapshots, names[-1]) if names else None


def _checkpoint_config(repo_id: str):
    """The checkpoint's own `config.json`, read off disk. None if not cached."""
    snapshot = _snapshot_dir(repo_id)
    if snapshot is None:
        return None, None
    path = os.path.join(snapshot, "config.json")
    if not os.path.isfile(path):
        return snapshot, None
    with open(path, encoding="utf-8") as handle:
        return snapshot, json.load(handle)


def _safetensors_numel(path: str) -> int:
    """Total element count of every tensor in a safetensors file.

    Read from the file's own header via `get_slice`, so no tensor is
    materialised and the count is independent of what the model reports.
    """
    from safetensors import safe_open

    total = 0
    with safe_open(path, framework="pt") as handle:
        for name in handle.keys():
            total += int(math.prod(handle.get_slice(name).get_shape()))
    return total


def _weight_file(snapshot: str) -> str:
    for name in sorted(os.listdir(snapshot)):
        if name.endswith(".safetensors"):
            return os.path.join(snapshot, name)
    pytest.fail(f"no safetensors file in the cached snapshot {snapshot}")


# ---------------------------------------------------------------- network guard


def _forbid_network(monkeypatch):
    """Fail loudly on any fetch, and record every `from_pretrained` attempt.

    The two `from_pretrained` entry points the loader uses are replaced with
    wrappers that raise unless the call is explicitly local-only. That is what
    makes "no request was made" checkable from inside the test instead of being
    inferred from a fast failure.

    Returns ``(calls, hub_attempts)`` where ``calls`` is every
    ``(class, local_only)`` attempt and ``hub_attempts`` is every direct Hub
    fetch. A gated refusal must leave both empty.
    """
    import transformers

    calls = []
    hub_attempts = []

    def _hub(*args, **kwargs):
        hub_attempts.append((args, sorted(kwargs)))
        raise AssertionError(
            "a HuggingFace Hub fetch was attempted where none is permitted")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _hub)
    monkeypatch.setattr("huggingface_hub.snapshot_download", _hub)

    for cls in (transformers.AutoTokenizer, transformers.AutoModelForCausalLM):
        real = cls.from_pretrained

        def _guarded(*args, __real=real, __cls=cls, **kwargs):
            local_only = kwargs.get("local_files_only") is True
            calls.append((__cls.__name__, local_only))
            if not local_only:
                raise AssertionError(
                    f"{__cls.__name__}.from_pretrained was called without "
                    f"local_files_only=True, so this call would have touched "
                    f"the network")
            return __real(*args, **kwargs)

        monkeypatch.setattr(cls, "from_pretrained", _guarded)

    return calls, hub_attempts


def _no_token(monkeypatch):
    """Remove every Hub token so the gated branch is the one under test."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- the predicate


def test_is_small_model_recognises_registry_entries_and_aliases():
    for name in ("smollm2-135m", "qwen2.5-0.5b", "llama-3.2-1b", "pythia-70m",
                 "  SMOLLM2-135M  ", "qwen2.5-0.5b-instruct"):
        assert svc.is_small_model(name) is True, name


@pytest.mark.parametrize("name", NON_SMALL_NAMES)
def test_is_small_model_rejects_everything_else(name):
    """A False is the fall-through signal; a wrong True reroutes real weights."""
    assert svc.is_small_model(name) is False, name


def test_out_of_registry_names_raise_keyerror_so_the_dispatcher_can_fall_through():
    for name in ("gpt2-small", "gemma-2b", "not-a-model"):
        with pytest.raises(KeyError) as caught:
            svc.load_model_request({"model_name": name})
        assert name in str(caught.value), (
            "the KeyError must name the request, so a caller can see which name "
            "was not a small model")


def test_the_default_model_name_is_gpt2_small_and_is_not_a_small_model():
    """`payload.get("model_name", "gpt2-small")` -- the default must not match."""
    assert svc.is_small_model("gpt2-small") is False
    with pytest.raises(KeyError):
        svc.load_model_request({})


# ---------------------------------------------------------------- the catalogue


def test_catalog_covers_every_registry_entry_without_claiming_live():
    rows = svc.catalog_entries()

    assert len(rows) == len(small_models.SMALL_MODELS), (
        "the catalogue must describe the whole registry, not a subset")
    assert {row["model_id"] for row in rows} == set(small_models.SMALL_MODELS)

    for row in rows:
        # Nothing in a catalogue listing executed a model, so `live` would be a
        # claim about a measurement that never happened.
        assert row["provenance"] != "live", row["model_id"]
        assert row["provenance"] == "reference", row["model_id"]


def test_catalog_distinguishes_a_reference_only_name_from_a_loadable_one():
    rows = {row["model_id"]: row for row in svc.catalog_entries()}

    for model_id, row in rows.items():
        cached, _ = small_models.local_weights_present(
            small_models.SMALL_MODELS[model_id])
        if row["loadable_now"]:
            assert cached, f"{model_id}: loadable_now requires local weights"
            assert row["load_state"] == "local-weights-cached"
        else:
            assert row["load_state"] != "local-weights-cached"
            assert row["load_state"] in ("not-downloaded", "gated"), row
        # The distinction has to be readable without re-deriving it.
        assert row["load_state"] in ("local-weights-cached", "not-downloaded",
                                     "gated")
        assert row["status"], model_id


def test_catalog_marks_every_field_it_reports():
    """A number a caller can quote must say where it came from.

    Only the numeric-coverage half of the provenance contract applies to a
    catalogue row: `is_derived_from_unmeasured` asks "is a value marked non-live
    being presented as measured", and a listing is *entirely* non-live by
    construction, so applying it here would flag the honest label.
    """
    from _provenance import assert_all_numbers_marked

    for row in svc.catalog_entries():
        assert set(row["field_provenance"]) == {
            k for k in row if k not in ("provenance", "field_provenance")}, (
                f"{row['model_id']}: field_provenance does not cover its fields")
        assert_all_numbers_marked(row, what=f"catalogue row {row['model_id']}")


def test_a_gated_row_says_gated_rather_than_not_downloaded():
    rows = {row["model_id"]: row for row in svc.catalog_entries()}
    if "llama-3.2-1b" not in rows:  # pragma: no cover - registry is pinned
        pytest.fail("llama-3.2-1b left the registry; this check is now vacuous")
    row = rows["llama-3.2-1b"]
    if "gated" in row["status"]:
        assert row["load_state"] == "gated"
        assert row["loadable_now"] is False


# ---------------------------------------------------------------- the live load


def _cached_or_skip(name: str) -> str:
    """A cached entry's repo id, or an explicit skip naming the missing state."""
    entry = small_models.SMALL_MODELS[name]
    cached, reason = small_models.local_weights_present(entry)
    if not cached:
        pytest.skip(f"{name} is not cached locally ({reason}); live load NOT RUN")
    return entry.hf_repo_id


@pytest.mark.parametrize("name", CACHED_CANDIDATES)
def test_a_cached_small_model_loads_live_with_config_reconciled_numbers(name):
    """The load is real, and its numbers are the checkpoint's own.

    The architecture below is compared against `config.json` read independently
    off the cache, and the parameter count against the safetensors header. A
    payload that merely looked plausible would not survive either comparison.
    """
    repo_id = _cached_or_skip(name)
    snapshot, config = _checkpoint_config(repo_id)
    if config is None:  # pragma: no cover - a weight file implies a config
        pytest.skip(f"{repo_id} has weights cached but no config.json")

    result = svc.load_model_request(
        {"model_name": name, "allow_download": False})

    assert result["status"] == "loaded", result.get("reason")
    assert result["provenance"] == "live"
    assert result["attested"] is True, (
        "a completed load must be attested by the measurement boundary")

    # Identity: which weights, not which name was typed.
    assert result["model_name"] == name
    assert result["hf_repo_id"] == repo_id
    assert result["model_loaded"] == name
    assert result["small_model"] is True

    # The architecture, reconciled against the checkpoint's own config.
    assert result["n_layers"] == config["num_hidden_layers"]
    assert result["n_heads"] == config["num_attention_heads"]
    assert result["d_model"] == config["hidden_size"]
    assert result["d_mlp"] == config["intermediate_size"]
    assert result["vocab_size"] == config["vocab_size"]
    assert result["context_length"] == config["max_position_embeddings"]
    assert result["n_kv_heads"] == config.get("num_key_value_heads")

    # And the same seven numbers the registry claims agree, or the discrepancy
    # is carried rather than dropped.
    if result["shape_disagreements"]:
        for row in result["shape_disagreements"]:
            assert "registry=" in row and "config=" in row, row

    # Weight accounting, against the file's own tensor shapes.
    assert result["n_parameters"] == _safetensors_numel(
        _weight_file(snapshot)), (
        "the reported parameter count is not the checkpoint's tensor count")
    assert result["weight_bytes"] >= result["n_parameters"] * 2, (
        "16-bit weights cannot be smaller than 2 bytes per parameter")
    assert result["device"] == "cpu", (
        "this box is shared; the loader's documented default is CPU")


@pytest.mark.parametrize("name", CACHED_CANDIDATES)
def test_a_live_payload_marks_every_number_it_reports(name):
    from _provenance import assert_all_numbers_marked, is_derived_from_unmeasured

    _cached_or_skip(name)
    result = svc.load_model_request(
        {"model_name": name, "allow_download": False})

    assert_all_numbers_marked(result, what="live small-model load")
    assert is_derived_from_unmeasured(result) == [], (
        "a live response carries a value marked non-live")
    for field in ("n_layers", "n_heads", "d_model", "d_mlp", "vocab_size",
                  "context_length", "n_parameters", "weight_bytes"):
        assert result["field_provenance"].get(field) == "live", field


# ------------------------------------------------------------- the refusals


def _first_uncached_ungated():
    for row in small_models.available_small_models():
        if not row["gated"] and not row["locally_cached"]:
            return small_models.SMALL_MODELS[row["model_id"]]
    return None


def test_an_uncached_model_without_permission_is_unavailable_not_placeholder(
        monkeypatch):
    """The refusal must name the repo and the switch, and carry no numbers."""
    entry = _first_uncached_ungated()
    if entry is None:
        pytest.skip("every non-gated registry entry is cached; NOT RUN")

    calls, hub_attempts = _forbid_network(monkeypatch)
    monkeypatch.delenv(DOWNLOAD_ENV, raising=False)

    result = svc.load_model_request(
        {"model_name": entry.model_id, "allow_download": False})

    assert result["status"] == "unavailable"
    assert result["provenance"] == "unavailable"
    assert result["attested"] is False
    assert entry.hf_repo_id in result["reason"], (
        "the reason must name the repository it could not read")
    assert DOWNLOAD_ENV in result["reason"], (
        "the reason must name the switch that would permit the fetch")
    assert result["field_provenance"]["model_name"] == "unavailable"

    # Nothing that reads as a measurement.
    for key in ("n_layers", "n_heads", "d_model", "d_mlp", "vocab_size",
                "context_length", "n_parameters", "weight_bytes"):
        assert result.get(key) is None, f"a refusal smuggled {key}={result[key]!r}"

    # And it genuinely did not fetch.
    assert hub_attempts == [], hub_attempts
    assert all(local_only for _, local_only in calls), calls


def test_downloads_are_not_permitted_by_absence_of_the_switch(monkeypatch):
    """`allow_download` defaults to False; the env switch is the other seam."""
    entry = _first_uncached_ungated()
    if entry is None:
        pytest.skip("every non-gated registry entry is cached; NOT RUN")

    monkeypatch.delenv(DOWNLOAD_ENV, raising=False)
    calls, hub_attempts = _forbid_network(monkeypatch)

    result = svc.load_model_request({"model_name": entry.model_id})

    assert result["status"] == "unavailable"
    assert hub_attempts == [], hub_attempts
    assert all(local_only for _, local_only in calls), calls


def test_the_gated_entry_is_refused_without_any_request(monkeypatch):
    """`allow_download=True` must not turn the gate into a 401.

    The request is *never made*: the gate is the reason, so it is reported as
    the reason. Asserted by watching the two `from_pretrained` entry points --
    `calls` must be empty, which is stronger than "it failed fast".
    """
    entry = small_models.SMALL_MODELS["llama-3.2-1b"]
    _no_token(monkeypatch)
    calls, hub_attempts = _forbid_network(monkeypatch)

    # Downloads permitted on purpose: the gate must hold anyway.
    result = svc.load_model_request(
        {"model_name": "llama-3.2-1b", "allow_download": True})

    assert result["status"] == "unavailable"
    assert result["provenance"] == "unavailable"
    assert "gated" in result["reason"].lower(), result["reason"]
    assert entry.hf_repo_id in result["reason"]
    assert "token" in result["reason"].lower()
    assert calls == [], f"the gated repo was contacted: {calls}"
    assert hub_attempts == [], hub_attempts
    assert result["gated"] is True


def test_a_live_label_is_unobtainable_without_a_completed_load(monkeypatch):
    """The one invariant that makes `live` mean anything.

    The loader is made to fail the way a real machine fails, and the result must
    come back unavailable -- not live with a substituted model.
    """
    from backend.science.models.small_models import SmallModelUnavailable

    def _refuse(*args, **kwargs):
        raise SmallModelUnavailable("smollm2-135m", "stubbed: no weights here")

    monkeypatch.setattr(small_models, "load_small_model", _refuse)
    monkeypatch.setattr(svc, "load_small_model", _refuse)

    result = svc.load_model_request({"model_name": "smollm2-135m"})

    assert result["status"] == "unavailable"
    assert result["provenance"] == "unavailable"
    assert result["provenance"] != "live"
    assert result["attested"] is False
    assert result["reason"] == "stubbed: no weights here", (
        "the refusal must be the loader's reason verbatim, not a rewrite")
    assert "n_layers" not in result or result["n_layers"] is None
    assert "measurement_record" not in result, (
        "an unattested refusal must carry no measurement record")


def test_a_non_small_model_never_reaches_the_small_model_loader(monkeypatch):
    """KeyError must be raised *before* any load is attempted."""
    from backend.science.models import small_models as mod

    def _explode(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("the small-model loader ran for gpt2-small")

    monkeypatch.setattr(mod, "load_small_model", _explode)
    monkeypatch.setattr(svc, "load_small_model", _explode)

    with pytest.raises(KeyError):
        svc.load_model_request({"model_name": "gpt2-small"})


def test_an_empty_payload_reports_a_reason_rather_than_a_model():
    """A refusal with no reason is a shrug; every path here must explain."""
    for name in ("smollm2-360m", "pythia-160m"):
        if not svc.is_small_model(name):  # pragma: no cover - registry is pinned
            continue
        result = svc.load_model_request(
            {"model_name": name, "allow_download": False})
        if result["status"] == "unavailable":
            assert result["reason"], name
            assert result["provenance"] == "unavailable"
        else:
            # Cached: then it really loaded and must say so.
            assert result["provenance"] == "live"
            assert result["attested"] is True
