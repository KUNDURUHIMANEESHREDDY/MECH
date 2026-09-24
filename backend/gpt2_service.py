"""Legacy GPT-2 service shim (FROZEN — do not extend).

``backend/gpt2_service.py`` never existed in this tree: ``legacy_dispatcher.py``
imports it in a ``try/except`` and every ``gpt2/*`` legacy route degraded to
``{"status": "error", "error": "gpp2_service unavailable: ..."}``.

These thin wrappers delegate to ``backend.services.gpt2_engine`` (the
survivor per LEGACY_DISPATCHER_AUDIT) using the legacy dict-payload calling
convention: each function takes the route payload dict ``p``.

New code must call ``backend.services.gpt2_engine`` (or the Society
``Executor`` agent) directly — never this module.
"""

from __future__ import annotations

from typing import Any, Dict

DEFAULT_PROMPT = "The capital of France is"


def _engine():  # type: ignore[no-untyped-def]
    from backend.services import gpt2_engine
    return gpt2_engine


def _unavailable() -> Dict[str, Any]:
    return {"status": "error",
            "error": "torch/transformers not available — cannot load real GPT-2"}


def load_model(p: Dict[str, Any]) -> Dict[str, Any]:
    engine = _engine()
    if not engine.is_available():
        return _unavailable()
    res = engine.load()
    res.setdefault("model_name", p.get("model_name", "gpt2"))
    return res


def run_prompt(p: Dict[str, Any]) -> Dict[str, Any]:
    engine = _engine()
    if not engine.is_available():
        return _unavailable()
    return engine.run_prompt(p.get("prompt") or DEFAULT_PROMPT)


def get_activations(p: Dict[str, Any]) -> Dict[str, Any]:
    engine = _engine()
    if not engine.is_available():
        return _unavailable()
    return engine.activations(int(p.get("layer", 0)))


def get_attention_head(p: Dict[str, Any]) -> Dict[str, Any]:
    engine = _engine()
    if not engine.is_available():
        return _unavailable()
    return engine.attention_head(int(p.get("layer", 0)),
                                 int(p.get("head", 0)))


def patch_head(p: Dict[str, Any]) -> Dict[str, Any]:
    engine = _engine()
    if not engine.is_available():
        return _unavailable()
    # patch_head reads _cache["prompt"] — populate it first.
    engine.run_prompt(p.get("prompt") or DEFAULT_PROMPT)
    return engine.patch_head(
        int(p.get("layer", 9)),
        int(p.get("head", 9)),
        p.get("pos_token", " Paris"),
        p.get("neg_token", " London"),
    )


def run_ioi(p: Dict[str, Any]) -> Dict[str, Any]:
    engine = _engine()
    if not engine.is_available():
        return _unavailable()
    return engine.ioi(p.get("io_name") or "John",
                      p.get("subj_name") or "Mary")
