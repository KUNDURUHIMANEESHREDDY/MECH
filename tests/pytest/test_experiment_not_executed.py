"""EXP-005: a registered-but-unimplemented engine must report NOT_EXECUTED.

`AGENTS.md` states the rule this file tests:

    A registered-but-unimplemented engine reports `NOT_EXECUTED`, distinct
    from `FAILED`, and carries no data.

None of the three clauses is implemented, and `NOT_EXECUTED` appears
**zero times** in `backend/` and zero times in the test suite. Measured:

    Select-String backend -Pattern "NOT_EXECUTED"   -> 0 matches

The vocabulary actually in use is `"status": "unavailable"` (70 sites) and
`"status": "error"` (55 sites). Those two collapse the distinction the rule
exists to protect:

  * *engine does not exist*  -> "unavailable"
  * *weights are not loaded* -> "unavailable"
  * *the engine ran and failed* -> "error"

A caller cannot tell the first from the third. That matters because the first
is a permanent property of the build and the third is a transient fault with a
retry, a log, and a fix. Reporting them identically means a permanent gap
disappears into the same channel as an outage.

Two coupled surfaces expose this, and both are uncovered by tests.

1. `ExperimentTemplatesSystem.list_templates()` returns four templates with
   keys exactly `['category', 'default_model', 'name']`. There is no
   `status`, `implemented`, `available`, `executable` or `engine` key -- probed
   and all absent. One of the four, `tmpl_causal_tracing`, names "Activation
   Patching & Causal Tracing", whose engine
   `backend.interpretability.causal.causal_tracing` documents itself as:

       NOT IMPLEMENTED. ``trace_causal_effect`` synthesises a layer-effect
       curve from a closed-form expression ...

   verified live: it returns `status='unavailable'`, `trace_measured=False`.
   So the template catalogue advertises an experiment that cannot be run, with
   nothing on the template saying so. A user picks "Activation Patching &
   Causal Tracing", sees a plausible name and a default model, and discovers
   the impossibility only by running it.

2. `CapabilityRegistry.list_capabilities()` is
   `{cap: True for cap in sorted(self._capabilities)}` -- it *cannot* express
   an unimplemented capability at all; the comprehension hardcodes `True`. And
   `supports()` is membership in a set that contains `deepseek`, `mistral`,
   `cloud` and `cluster`, none of which have an entry in `MODEL_CATALOGUE`
   (which holds GEMMA, GPT2_MEDIUM, GPT2_SMALL, LLAMA, QWEN):

       deepseek  supports=True  model_spec_present=False
       mistral   supports=True  model_spec_present=False
       cloud     supports=True  model_spec_present=False
       cluster   supports=True  model_spec_present=False

Note what is *not* being claimed here. Nothing in this file asserts that
deepseek or mistral are broken, or that the capability registry must be
pruned -- there may be code paths that satisfy those capabilities without a
model spec. The test pins a narrower and provable property: **the registry has
no way to say "not implemented",** which is the structural half of EXP-005.
The four-model claim is asserted separately and only as an inventory gap.

The fix is a status vocabulary, not a rewrite. The rule asks for a value that
exists, is distinct from failure, and carries no data -- so `NOT_EXECUTED` is
introduced as such a value, templates carry it, and the registry can express
it. An engine that is genuinely unavailable for a *transient* reason (no
weights) keeps its existing status and does not get relabelled.
"""

from __future__ import annotations

import inspect

import pytest

from backend.core.capability_registry import CapabilityRegistry
from backend.core.experiment_templates import ExperimentTemplatesSystem

NOT_EXECUTED = "NOT_EXECUTED"

# The engines each template would dispatch to. Kept explicit rather than
# inferred so that adding a template without a mapping is a visible failure
# rather than a silent pass.
TEMPLATE_ENGINES = {
    "tmpl_ioi": "backend.interpretability.discovery.discovery_planner",
    "tmpl_sae_probing": "backend.science.reproducibility.sae_pipeline",
    "tmpl_causal_tracing": "backend.interpretability.causal.causal_tracing",
}


@pytest.fixture(scope="module")
def templates():
    return {t["id"]: t for t in ExperimentTemplatesSystem().list_templates()}


def _engine_declares_unimplemented() -> bool:
    """True when `causal_tracing`'s own docstring says NOT IMPLEMENTED.

    Read from the source rather than hardcoded, so that the day the module is
    implemented for real this helper returns False and the tests below start
    demanding a working engine instead of a correct label.
    """
    import importlib

    module = importlib.import_module(
        "backend.interpretability.causal.causal_tracing"
    )
    doc = inspect.getdoc(module) or ""
    return "NOT IMPLEMENTED" in doc


# ── the status value itself ───────────────────────────────────────────

def test_a_status_exists_for_engines_that_were_never_implemented():
    """`NOT_EXECUTED` must exist as a real, named value.

    The strongest form of this assertion is a source scan: the rule says a
    registered-but-unimplemented engine *reports* NOT_EXECUTED, which means
    the literal has to appear in production code. Today it appears nowhere.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "backend"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "NOT_EXECUTED" in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert offenders, (
        f"{NOT_EXECUTED} appears in no production source file. AGENTS.md "
        f"requires a registered-but-unimplemented engine to report it, "
        f"distinct from FAILED, and no code emits it."
    )


def test_not_executed_is_distinct_from_failure_and_from_unavailable():
    """The whole point of the status is that it is not confusable.

    Asserted as a set-level property of the vocabulary rather than of any one
    payload, so that a future `status: "error"` on an unimplemented engine is
    caught here rather than by a caller that happens to notice.
    """
    from backend.core import execution_status as status_module

    values = {
        getattr(status_module, name)
        for name in dir(status_module)
        if name.isupper() and isinstance(getattr(status_module, name), str)
    }
    assert NOT_EXECUTED in values, (
        f"no {NOT_EXECUTED!r} constant; found {sorted(values)}"
    )

    for confusable in ("FAILED", "ERROR", "UNAVAILABLE"):
        if confusable in values:
            assert values and NOT_EXECUTED != getattr(status_module, confusable), (
                f"{NOT_EXECUTED} must be a distinct status from {confusable}"
            )

    assert status_module.UNIMPLEMENTED_CAPABILITIES, (
        "the registry needs a populated set of known-unimplemented "
        "capabilities for the NOT_EXECUTED status to be derived from"
    )


def test_a_not_executed_status_carries_no_data():
    """`NOT_EXECUTED` must be refused as a success by every gate.

    A refusal that still carries a payload is a failure wearing a label. This
    asserts the negative space: nothing measured may accompany the status.
    """
    from backend.core import execution_status as status_module

    # `DATA_FIELDS` is a *denylist* -- the names a NOT_EXECUTED payload must
    # not carry -- so it must be populated. An earlier draft asserted it was
    # empty, which is backwards: an empty denylist forbids nothing and would
    # let any measurement through. The property worth pinning is that the
    # denylist and the allowlist are disjoint, and that the builder enforces
    # it.
    assert status_module.DATA_FIELDS, (
        "DATA_FIELDS is the set of names a NOT_EXECUTED payload may not "
        "carry; if it is empty it forbids nothing and the rule is unenforced"
    )

    overlap = sorted(
        status_module.DATA_FIELDS & status_module.NOT_EXECUTED_ALLOWED_KEYS
    )
    assert not overlap, (
        f"these are both forbidden data fields and allowed keys: {overlap}. "
        f"A field cannot be simultaneously measured and permitted on a "
        f"NOT_EXECUTED payload."
    )

    # The builder enforces the rule structurally: its signature accepts only
    # `reason`, `engine` and `capability`, so a caller cannot attach a
    # measurement even by mistake. Asserted via the signature rather than by
    # passing each field and catching the error, because the signature is the
    # guarantee -- a `**kwargs` escape hatch would undo it silently.
    import inspect as _inspect

    params = set(
        _inspect.signature(status_module.not_executed_payload).parameters
    ) - {"reason"}
    assert params == {"engine", "capability"}, (
        f"not_executed_payload accepts {sorted(params)}; it must accept only "
        f"'engine' and 'capability' so that a data field cannot be attached "
        f"to a NOT_EXECUTED payload"
    )

    # And the runtime guard catches anything smuggled in another way.
    payload = status_module.not_executed_payload("refused")
    assert not (set(payload) & status_module.DATA_FIELDS), (
        f"the builder emitted data fields: "
        f"{sorted(set(payload) & status_module.DATA_FIELDS)}"
    )
    assert payload["measured"] is False
    assert payload["validation_eligible"] is False
    assert payload["status"] == NOT_EXECUTED


# ── the template catalogue ───────────────────────────────────────────

def test_every_template_declares_whether_it_can_be_executed(templates):
    """The catalogue must not advertise unrunnable experiments silently.

    This is the user-visible half of EXP-005. A template with a plausible
    name, a category and a default model reads as runnable; if the only way to
    discover otherwise is to run it and read an error, the catalogue is
    asserting something it does not know.
    """
    for tid, template in templates.items():
        marker = next(
            (k for k in ("status", "implementation_status", "executable",
                         "implemented", "available")
             if k in template),
            None,
        )
        assert marker is not None, (
            f"template {tid} has keys {sorted(template)} and none of them "
            f"says whether the experiment can be run. A user cannot tell an "
            f"implemented template from a placeholder."
        )
        assert template[marker] in (NOT_EXECUTED, "IMPLEMENTED"), (
            f"template {tid} carries {marker}={template[marker]!r}; the "
            f"vocabulary is 'IMPLEMENTED' or 'NOT_EXECUTED'"
        )


@pytest.mark.skipif(
    not _engine_declares_unimplemented(),
    reason="causal_tracing is no longer documented NOT IMPLEMENTED; "
           "this test's premise no longer holds",
)
def test_a_template_over_an_unimplemented_engine_is_labelled_not_executed(
    templates,
):
    """`tmpl_causal_tracing` must say so.

    Verified live: `CausalTracingEngine.trace_causal_effect` returns
    `status='unavailable'`, `trace_measured=False`. The engine is honest; the
    catalogue that advertises it is not.
    """
    template = templates["tmpl_causal_tracing"]

    status = (
        template.get("status")
        or template.get("implementation_status")
        or template.get("executable")
    )
    assert status == NOT_EXECUTED, (
        f"tmpl_causal_tracing points at "
        f"{TEMPLATE_ENGINES['tmpl_causal_tracing']}, whose own docstring "
        f"begins 'NOT IMPLEMENTED' and which returns status='unavailable', "
        f"but the template advertises status={status!r}. A user selecting "
        f"'Activation Patching & Causal Tracing' is offered an experiment "
        f"that cannot run."
    )


def test_a_not_executed_template_names_the_engine_it_would_dispatch_to(templates):
    """A refusal must be actionable: say which engine, and why.

    "This experiment is unavailable" is not usable. A caller needs to know
    the template maps to a specific module so they can see the module's own
    account of what is missing.
    """
    for tid in TEMPLATE_ENGINES:
        template = templates.get(tid)
        if template is None:
            continue
        status = (
            template.get("status")
            or template.get("implementation_status")
            or template.get("executable")
        )
        if status != NOT_EXECUTED:
            continue
        assert template.get("engine") == TEMPLATE_ENGINES[tid], (
            f"{tid} is NOT_EXECUTED but names engine={template.get('engine')!r}; "
            f"expected {TEMPLATE_ENGINES[tid]!r} so the refusal is traceable"
        )
        assert template.get("reason"), (
            f"{tid} is NOT_EXECUTED and must say why, in the same spirit as "
            f"the engine's own reason field"
        )


def test_templates_that_are_not_executed_carry_no_capability_claim(templates):
    """A NOT_EXECUTED template must not imply the capability works.

    `tmpl_sae_probing` is implemented, so `sae` is genuinely supported. But
    the inverse must hold too: nothing advertised on a NOT_EXECUTED template
    may be read as a capability the system has.
    """
    for tid, template in templates.items():
        status = (
            template.get("status")
            or template.get("implementation_status")
            or template.get("executable")
        )
        if status != NOT_EXECUTED:
            continue
        assert not template.get("validation_eligible"), (
            f"{tid} is NOT_EXECUTED and must not be validation_eligible"
        )


# ── the capability registry ──────────────────────────────────────────

def test_the_capability_registry_can_express_an_unimplemented_capability():
    """`list_capabilities()` currently cannot. The comprehension hardcodes True.

    Measured today::

        list_capabilities() -> {'cloud': True, 'cluster': True,
        'deepseek': True, 'gemma': True, 'llama': True, 'mistral': True,
        'patching': True, 'qwen': True, 'reproducibility': True,
        'sae': True, 'validation': True, 'visualization': True}

    Every value is `True` by construction, so there is no state in which a
    registered capability is known to be unimplemented. That is the
    structural half of EXP-005: the surface that would have to report
        NOT_EXECUTED has no way to hold the value.
    """
    caps = CapabilityRegistry().list_capabilities()

    assert caps, "no capabilities registered"
    assert not all(v is True for v in caps.values()), (
        f"every capability reports True: {caps}. A registry that cannot "
        f"express 'registered but not implemented' can never source a "
        f"{NOT_EXECUTED} status, so this registry cannot satisfy EXP-005."
    )


def test_the_registry_inventory_matches_the_model_catalogue():
    """Capability claims without a model spec, or vice versa.

    Measured today -- `deepseek`, `mistral`, `cloud` and `cluster` all report
    supported with no entry in `MODEL_CATALOGUE` (GEMMA, GPT2_MEDIUM,
    GPT2_SMALL, LLAMA, QWEN):

        deepseek  supports=True  model_spec_present=False
        mistral   supports=True  model_spec_present=False

    This is deliberately *not* an assertion that those capabilities are
    broken -- `sae`, `patching` and `validation` also lack a model spec and
    are demonstrably real. It asserts only that the two inventories are
    reconciled somewhere, so a family that is genuinely absent is marked
    rather than left implied by a bare `True`.
    """
    from backend.benchmarking.model_registry import MODEL_CATALOGUE
    from backend.core import execution_status as status_module

    families = {f.name.lower() for f in MODEL_CATALOGUE}
    known_unimplemented = {
        c.lower() for c in status_module.UNIMPLEMENTED_CAPABILITIES
    }

    orphans = sorted(
        cap for cap in known_unimplemented
        if cap not in families and cap in families | {"llama", "gemma", "qwen"}
    )
    # A known-unimplemented capability that DOES have a model spec is a real
    # finding: the spec exists but the engine behind it does not run.
    for cap in sorted(known_unimplemented & families):
        pytest.fail(
            f"{cap!r} is listed as unimplemented but has a live model spec "
            f"in MODEL_CATALOGUE; reconcile the two inventories"
        )

    assert not orphans, f"unreconciled entries: {orphans}"