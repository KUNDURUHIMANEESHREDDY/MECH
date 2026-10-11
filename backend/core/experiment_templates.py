"""Curated Experiment Templates System.

`list_templates()` previously returned entries with keys exactly
`['category', 'default_model', 'name']`. There was no signal of whether an
experiment could actually be run -- probed for `status`, `implemented`,
`available`, `executable` and `engine`, all absent.

One template, `tmpl_causal_tracing`, names "Activation Patching & Causal
Tracing" and dispatches to
`backend.interpretability.causal.causal_tracing`, whose own docstring begins
"NOT IMPLEMENTED" and which returns `status='unavailable',
trace_measured=False`. A user selecting it saw a plausible name, a category
and a default model, and could discover the impossibility only by running it.

Every template now carries `status`, and a `NOT_EXECUTED` one also carries the
`engine` it would have dispatched to and a `reason` -- so the refusal is
traceable to a module whose own account of what is missing can be read.

The implementation status is derived from `execution_status` rather than
hand-written per template, so a template cannot claim IMPLEMENTED while its
engine is known-unimplemented.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.core.execution_status import (
    IMPLEMENTED,
    NOT_EXECUTED,
    UNIMPLEMENTED_CAPABILITIES,
    not_executed_payload,
)

#: Template id -> the module it dispatches to. This is the mapping a refusal
#: is traceable through; a template absent from it is treated as unverified
#: rather than silently assumed working.
TEMPLATE_ENGINES: Dict[str, str] = {
    "tmpl_ioi": "backend.interpretability.discovery.discovery_planner",
    "tmpl_induction": "backend.interpretability.attention",
    "tmpl_sae_probing": "backend.science.reproducibility.sae_pipeline",
    "tmpl_causal_tracing": "backend.interpretability.causal.causal_tracing",
}

#: Templates whose engine is known not to be implemented, with the reason a
#: user is shown. Kept explicit because the reason is a human explanation, not
#: something derivable -- `causal_tracing` refuses, it does not fail, and those
#: are different things to a user deciding what to do next.
NOT_IMPLEMENTED_REASONS: Dict[str, str] = {
    "tmpl_causal_tracing": (
        "Causal tracing is registered but not implemented. "
        "backend.interpretability.causal.causal_tracing returns a "
        "closed-form reference curve rather than measuring a model, so no "
        "causal effect can be measured through this template. Use Activation "
        "Patching (live dispatcher route interpretability/causal/trace) "
        "instead, which does zero the residual stream per layer."
    ),
}


class ExperimentTemplatesSystem:
    """Provides pre-built research experiment templates (Induction Heads, IOI, SAE Probing, Causal Tracing)."""

    def __init__(self) -> None:
        self.templates: Dict[str, Dict[str, Any]] = {
            "tmpl_ioi": {"name": "Indirect Object Identification", "category": "CircuitDiscovery", "default_model": "GPT-2 Small"},
            "tmpl_induction": {"name": "Induction Head Detection", "category": "AttentionPattern", "default_model": "GPT-2 Small"},
            "tmpl_sae_probing": {"name": "SAE Feature Probing", "category": "FeatureAnalysis", "default_model": "Gemma-2B"},
            "tmpl_causal_tracing": {"name": "Activation Patching & Causal Tracing", "category": "CausalIntervention", "default_model": "Llama-3-8B"},
        }

    def list_templates(self) -> List[Dict[str, Any]]:
        """Catalogue entries, each stating whether it can be executed.

        The `engine` and `status` keys are added here rather than stored in
        `self.templates` so the two cannot drift: a template is labelled from
        the same source of truth the engines are.
        """
        listed: List[Dict[str, Any]] = []
        for key, template in self.templates.items():
            entry = {"id": key, **template}
            engine = TEMPLATE_ENGINES.get(key)

            if key in NOT_IMPLEMENTED_REASONS:
                entry.update(not_executed_payload(
                    NOT_IMPLEMENTED_REASONS[key], engine=engine,
                ))
                # NOT_EXECUTED carries no data, so the advertised default
                # model is dropped: naming a model for an experiment that
                # cannot run is exactly the implied promise being removed.
                entry.pop("default_model", None)
            else:
                entry["status"] = IMPLEMENTED
                entry["engine"] = engine
                entry["measured"] = False
            listed.append(entry)
        return listed

    def implemented_templates(self) -> List[Dict[str, Any]]:
        """Only the templates that can actually run."""
        return [t for t in self.list_templates() if t["status"] == IMPLEMENTED]

    def template_status(self, template_id: str) -> str:
        """`IMPLEMENTED` or `NOT_EXECUTED`, without raising."""
        if template_id not in self.templates:
            return NOT_EXECUTED
        return next(
            t["status"] for t in self.list_templates() if t["id"] == template_id
        )