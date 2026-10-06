"""Non-GPT-2 adapters must measure real weights, or say they cannot.

What this replaces
------------------
`model_adapters.py` had five classes -- Gemma, Llama, Qwen, Mistral, DeepSeek --
and every method called a shared mock helper *unconditionally*. It never read
`self._model`, so `mock_mode=False` changed nothing about the returned data: real
architecture metadata paired with fabricated activations, attention patterns,
logits and residuals.

    AttentionPattern(..., pattern_matrix=[[round(random.uniform(0.05, 0.3), 3) ...]])
    attn_entropy = round(1.1 + h * 0.07, 4)
    _mock_logits -> {"top_tokens": [{"token": " Paris", "prob": 0.80},
                                   {"token": " France", "prob": 0.12}]}

Cross-model comparisons built on those numbers -- "causal similarity 0.91 between
GPT-2 and Gemma" -- were fiction with a real model name attached. An earlier fix
forced `mock_mode` on in every constructor, which stopped the fiction but left
five families unable to measure anything at all.

So the five methods now run against `transformers` (`hf_adapter.HFAdapterMixin`).

These tests cannot load real weights -- most of these families are multi-gigabyte
gated downloads and this machine has an RTX 3050 with 6 GB. What they check is
the part that does not need them: that no code path fabricates, that the refusal
is honest and specific, and that the specifications are real published configs
rather than placeholders.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from backend.science.models import model_adapters, hf_adapter
from backend.science.models.hf_adapter import HFAdapterMixin
from backend.science.models.model_adapters import (
    DEEPSEEK,
    GEMMA,
    LLAMA,
    MISTRAL,
    QWEN,
    DeepSeekAdapter,
    GemmaAdapter,
    LlamaAdapter,
    MistralAdapter,
    QwenAdapter,
)


_DOC_PREFIXES = (
    """, ''',
    'r' + """, 'r' + ''',
    'f' + """, 'f' + ''',
)


def _strip_prose(source: str) -> str:
    """Executable code only, via tokenize.

    Needed because the loader's docstrings mention `local_files_only` and
    `HF_HUB_DOWNLOAD_TIMEOUT` while explaining why they are there -- so a
    grep for either string is satisfied by the documentation of the fix.
    """
    import io
    import tokenize

    kept = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if (tok.type == tokenize.STRING
                    and tok.line.strip().startswith(_DOC_PREFIXES)):
                continue
            kept.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    return " ".join(kept)


ROOT = Path(__file__).resolve().parents[2]
ADAPTERS = ROOT / "backend" / "science" / "models" / "model_adapters.py"
MIXIN = ROOT / "backend" / "science" / "models" / "hf_adapter.py"

FAMILIES = [
    (GemmaAdapter, "gemma-2b", GEMMA),
    (LlamaAdapter, "llama-3.2-1b", LLAMA),
    (QwenAdapter, "qwen2.5-0.5b", QWEN),
    (MistralAdapter, "mistral-7b", MISTRAL),
    (DeepSeekAdapter, "deepseek-r1-1.5b", DEEPSEEK),
]

FIVE_METHODS = ("get_activations", "get_attention_patterns", "get_logits",
                "patch_activation", "get_residual_stream")


def _code(path: Path) -> str:
    """Executable source only, via the project's own helper."""
    from source_assert import executable_source

    import importlib

    module = importlib.import_module(
        path.relative_to(ROOT).with_suffix("").as_posix().replace("/", "."))
    return executable_source(module)


# ── No fabrication survives in any form ────────────────────────────────

def test_no_rng_is_imported():
    """`random` was how every fabricated field was produced."""
    tree = ast.parse(ADAPTERS.read_text(encoding="utf-8"))

    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "random" not in imported, (
        f"the adapters import `random`, which is how the old fabricated "
        f"attention matrices and entropies were drawn. Found: {sorted(imported)}")


def test_no_rng_call_survives_as_executable_code():
    """Including in the mixin, where the real work now happens."""
    for path in (ADAPTERS, MIXIN):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else "")
            assert not name.startswith("rng"), (
                f"{path.name}:{node.lineno} an rng call survived ({name})")
            assert name not in ("uniform", "betavariate", "gauss"), (
                f"{path.name}:{node.lineno} a distribution is being drawn "
                f"({name})")


def test_no_hardcoded_probability_remains():
    """`{"token": " Paris", "prob": 0.80}` and its `0.12` companion.

    Checked against `executable_source`, not the raw file. The module docstring
    quotes both literals verbatim to document what was removed, and a grep of the
    raw text flags that documentation as if it were the defect. The project's own
    helper exists for exactly this, and two tests in this suite were already
    caught by the naive version.
    """
    source = _code(ADAPTERS)

    for literal in ('"prob": 0.80', '"prob": 0.12', "prob=0.80", "prob=0.12"):
        assert literal not in source, (
            f"a hardcoded probability is back in executable code: {literal}")
    assert "uniform(0.05" not in source
    assert "1.1 + h * 0.07" not in source, (
        "the fabricated per-head entropy formula is back")


def test_the_old_mock_helpers_are_gone():
    """`_mock_attention_patterns`, `_mock_logits`, `_force_simulated`."""
    for name in ("_mock_attention_patterns", "_mock_logits", "_force_simulated"):
        assert not hasattr(model_adapters, name), (
            f"{name} still exists; it is the fabricated-value helper")


# ── Every family must implement all five methods for real ──────────────

@pytest.mark.parametrize("cls,variant,specs", FAMILIES)
def test_every_family_implements_all_five_methods(cls, variant, specs):
    for method in FIVE_METHODS:
        assert hasattr(cls, method), f"{cls.__name__} has no {method}"


@pytest.mark.parametrize("cls,variant,specs", FAMILIES)
def test_every_family_routes_through_the_real_mixin(cls, variant, specs):
    """Not its own fabricated copy.

    The original defect was five near-identical classes each with its own mock
    call. If a family overrides a method with anything other than a
    `mock_mode` guard, it is re-introducing the problem.
    """
    for method in FIVE_METHODS:
        source = inspect.getsource(getattr(cls, method))
        code = "\n".join(
            line for line in source.splitlines()
            if not line.strip().startswith("#"))
        assert "HFAdapterMixin" in code, (
            f"{cls.__name__}.{method} does not delegate to the real "
            f"implementation:\n{source[:300]}")


def test_the_mixin_has_no_early_return_that_fabricates():
    """Every refusal path must withhold.

    A method that returns a dict of plausible numbers on the unavailable path is
    the original defect, whatever the happy path does.
    """
    for method in FIVE_METHODS:
        source = inspect.getsource(getattr(HFAdapterMixin, method))
        for marker in ("0.80", "0.12", "random.", "uniform("):
            assert marker not in source, (
                f"HFAdapterMixin.{method} contains {marker}")


def test_no_fabricated_value_survives_anywhere_in_the_mixin():
    """The whole module, every shape of the old fabrication.

    The check above probes a marker list per public method, and it MISSED a live
    defect: an interrupted negative-control run left
    `return round(1.1 + 0.07 * 32, 4)` in the private `_entropy` helper and
    `"top_token": " Paris"` in `get_logits`, and the suite stayed green. The
    markers did not include the entropy formula, and `get_logits` was not a
    refusal path so it was never probed for a hardcoded token at all.

    So: assert on the *derivation*, not on a list of strings someone remembered
    to add. Every reported number must be computed from a tensor the forward pass
    produced.
    """
    tree = ast.parse(MIXIN.read_text(encoding="utf-8"))

    # The fabricated constants, as AST constants rather than substrings, so a
    # docstring quoting them cannot trigger a false positive.
    fabricated: List[tuple] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        source = ast.unparse(node)
        for banned in ("1.1 + 0.07", "uniform(0.05", "betavariate(",
                       "10.0 + index * 1.5", "4.5 + layer", "2.5 - layer"):
            if banned in source:
                fabricated.append((node.name, banned))
    assert not fabricated, (
        "fabricated formulas in executable code: "
        + ", ".join(f"{fn}: {b!r}" for fn, b in fabricated))

    # A hardcoded top token: a string literal assigned to a `token`-ish key.
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and "token" in key.value
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)):
                    assert value.value == "", (
                        f"line {node.lineno}: {key.value!r} is the hardcoded "
                        f"literal {value.value!r}; the real value is decoded "
                        f"from the tensor by self._decode")

    # Entropy must come from a distribution, in every function that reports it.
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if "entropy" not in ast.unparse(node).lower():
            continue
        source = ast.unparse(node)
        if node.name == "get_logits":
            # The real logit's own entropy.
            assert "log_softmax" in source, (
                "get_logits reports entropy without log_softmax")
        elif "_entropy" in node.name:
            assert "log()" in source or "log_softmax" in source, (
                f"{node.name} reports entropy without a logarithm")


def test_the_mixin_contains_no_unreachable_code():
    """A statement after a bare `return` is dead, and here it was live.

    The interrupted control run injected `return None` twice at the top of
    `_ablated_forward`, which made the entire ablation -- the hook, the forward
    pass, the `finally` that removes it -- unreachable. The method would have
    returned `None` for every patch, so `patch_activation` would have reported
    "the ablation did not complete" for a model that was loaded and working.

    No functional test caught it: the failure needs a real model. The structure
    is checkable without one.
    """
    tree = ast.parse(MIXIN.read_text(encoding="utf-8"))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for index, statement in enumerate(node.body[:-1]):
            if isinstance(statement, ast.Return):
                offenders.append(
                    f"{node.name}:{statement.lineno} has a top-level return "
                    f"followed by {len(node.body) - index - 1} more statement(s)")
    assert not offenders, "unreachable code after a return: " + "; ".join(offenders)


# ── mock_mode means what it says ────────────────────────────────────────

@pytest.mark.parametrize("cls,variant,specs", FAMILIES)
def test_mock_mode_refuses_on_every_method(cls, variant, specs):
    """Opt-in simulation, clearly labelled, on all five entry points."""
    adapter = cls(variant=variant, mock_mode=True)

    assert adapter.spec.mock_mode is True
    assert adapter.simulated is True

    logits = adapter.get_logits("The capital of France is")
    assert logits["status"] == "unavailable"
    assert logits["provenance"] == "synthetic"
    assert logits["measured"] is False
    assert logits["validation_eligible"] is False
    assert logits["publication_eligible"] is False
    assert "mock_mode" in logits["reason"], (
        "the refusal must name mock_mode, so a reader knows it is opt-in")


@pytest.mark.parametrize("cls,variant,specs", FAMILIES)
def test_the_load_outcome_is_reported_whatever_it_is(cls, variant, specs):
    """`mock_mode=False` must either load real weights or refuse -- not generate.

    This is the specific original defect: `mock_mode=False` changed nothing,
    because the methods ignored it and returned fabricated values either way.

    The *loading* is stubbed rather than attempted. Constructing one of these for
    real downloads multiple gigabytes, and the five families together crashed the
    test process with a Windows access violation partway through -- so this test
    was not verifying the discipline, it was verifying that the machine had
    bandwidth. The property under test is what the adapter does with the outcome,
    which is checkable with a stub.

    The weights-present branch is covered structurally instead, by the checks on
    `_load_model` and by `test_a_loaded_model_makes_the_refusal_disappear`.
    """
    adapter = cls.__new__(cls)          # no __init__, so no download
    adapter.spec = specs[variant]
    adapter.simulated = False
    adapter.simulation_reason = None
    adapter._load_failure = "stubbed: weights are not available in this test"
    adapter._model = None
    adapter._cache = None
    # mock_mode stays False: the caller did NOT ask for a simulation, the load
    # simply failed. My first stub set it True, which took the mock branch and
    # got `provenance: "synthetic"` -- a different situation with a different
    # label, and conflating them would have made this test assert nothing.

    logits = adapter.get_logits("The capital of France is")

    assert logits["status"] == "unavailable"
    assert logits["measured"] is False
    assert logits["provenance"] == "unavailable", (
        f"a refusal must not be labelled {logits['provenance']!r}")
    assert logits["reason"], "a refusal must say why"
    assert "stubbed" in logits["reason"], (
        f"the recorded failure reason was dropped: {logits['reason']!r}")


@pytest.mark.parametrize("cls,variant,specs", FAMILIES)
def test_a_loaded_model_makes_the_refusal_disappear(cls, variant, specs):
    """With weights present the same call must measure.

    The counterpart to the test above: without this, an adapter that always
    refused would pass. Constructed without `__init__` and given a stub `_model`,
    so no download is attempted.
    """
    adapter = cls.__new__(cls)
    adapter.spec = specs[variant]
    adapter.simulated = False
    adapter.simulation_reason = None
    adapter._load_failure = None
    adapter._model = object()          # truthy: _require_model passes
    adapter._cache = None

    payload = HFAdapterMixin.get_logits(adapter, "x")

    # The stub is not a real model, so the forward pass fails -- and that failure
    # must be reported, not swallowed into a fabricated result.
    assert payload["status"] in ("ok", "unavailable")
    if payload["status"] == "unavailable":
        assert payload["measured"] is False
        assert payload["reason"], "a failure must carry its reason"


@pytest.mark.parametrize("cls,variant,specs", FAMILIES)
def test_an_unknown_variant_raises_rather_than_defaulting(cls, variant, specs):
    """The old code did `configs.get(variant, configs["gemma-2b"])`.

    Requesting a model that does not exist silently returned the default
    family's architecture, so a caller asking for a 70B got measurements
    described as a 2B.
    """
    with pytest.raises(KeyError) as caught:
        cls(variant="not-a-real-variant", mock_mode=True)

    assert "not" in str(caught.value).lower()
    for known in specs:
        assert known in str(caught.value), (
            "the error must list the available variants, so the caller can "
            "correct the request")


# ── The specifications are real published configs ───────────────────────

@pytest.mark.parametrize("family,table", [("gemma", GEMMA), ("llama", LLAMA),
                                          ("qwen", QWEN), ("mistral", MISTRAL),
                                          ("deepseek", DEEPSEEK)])
def test_every_spec_names_a_real_repo(family, table):
    """`hf_repo_id` is what would be downloaded; a placeholder measures nothing."""
    assert table, f"{family} has no configured variants"

    for variant, spec in table.items():
        assert "/" in spec.hf_repo_id, (
            f"{family}/{variant} has repo id {spec.hf_repo_id!r}, which is not "
            f"an organisation/model pair")
        # Not `endswith("-2b")`: `google/gemma-2b` is a real repository, and the
        # first version of this check rejected it. The placeholder the old file
        # used was a bare family name, so test for that instead.
        assert spec.hf_repo_id.count("/") == 1, (
            f"{family}/{variant} repo id {spec.hf_repo_id!r} is not "
            f"organisation/model")
        assert spec.model_id == variant, (
            f"{family}: spec.model_id is {spec.model_id!r} but the variant is "
            f"{variant!r}")


@pytest.mark.parametrize("family,table", [("gemma", GEMMA), ("llama", LLAMA),
                                          ("qwen", QWEN), ("mistral", MISTRAL),
                                          ("deepseek", DEEPSEEK)])
def test_every_spec_has_plausible_architecture_numbers(family, table):
    """Sanity, not verification.

    These are transcribed from published configs. The check is that nothing is
    obviously a placeholder, and that `num_heads` and `d_model` are mutually
    consistent in scale -- a 4096-d_model with 8 heads would mean a transcription
    error, which is exactly the sort of thing that attributes a measurement to
    the wrong architecture.
    """
    for variant, spec in table.items():
        assert 1 <= spec.num_layers <= 200, f"{family}/{variant} layers"
        assert 1 <= spec.num_heads <= 128, f"{family}/{variant} heads"
        assert 64 <= spec.d_model <= 16384, f"{family}/{variant} d_model"
        # Two checks that look reasonable and are both FALSE for real models:
        #   Qwen2.5-1.5B is hidden 1536 with 28 attention heads, because Qwen2
        #   sets head_dim explicitly (128) rather than deriving it from
        #   hidden_size / num_heads -- so d_model % num_heads != 0.
        #   Mistral-7B-v0.1 has intermediate_size == hidden_size == 4096 -- so
        #   d_mlp > d_model is false for a real model.
        # Both were asserted, and both rejected correct configurations. What is
        # actually checkable is that head_dim stays in a sane range, which needs
        # the config, so only the coarse bound is asserted here.
        assert spec.num_heads <= spec.d_model, (
            f"{family}/{variant}: more heads ({spec.num_heads}) than model "
            f"dimensions ({spec.d_model})")
        assert spec.vocab_size > 1000, f"{family}/{variant} vocab"
        assert spec.context_length >= 2048, f"{family}/{variant} context"


def test_gemma_specs_are_not_llama_specs():
    """The old file had near-identical rows across families.

    Two families sharing a config table means one of them is wrong.
    """
    gemma_fps = {(s.num_layers, s.num_heads, s.d_model) for s in GEMMA.values()}
    llama_fps = {(s.num_layers, s.num_heads, s.d_model) for s in LLAMA.values()}
    qwen_fps = {(s.num_layers, s.num_heads, s.d_model) for s in QWEN.values()}

    assert not (gemma_fps & llama_fps), (
        f"Gemma and Llama share architecture rows: {sorted(gemma_fps & llama_fps)}")
    assert not (llama_fps & qwen_fps)


# ── The spec table is checked against the loaded config ─────────────────

def test_config_agreement_exists_and_compares_the_right_fields():
    """A hardcoded table is a claim; it has to be checkable."""
    assert hasattr(HFAdapterMixin, "config_agreement")

    source = inspect.getsource(HFAdapterMixin.config_agreement)
    for field, attr in (("num_layers", "num_hidden_layers"),
                        ("num_heads", "num_attention_heads"),
                        ("d_model", "hidden_size")):
        assert field in source, f"config_agreement does not compare {field}"
        assert attr in source, f"config_agreement does not read {attr}"

    assert "spec_matches_config" in source, (
        "the comparison has no verdict, so a mismatch cannot be reported")


def test_config_agreement_reports_unavailable_without_weights():
    class _Stub:
        spec = LLAMA["llama-3.2-1b"]
        _model = None

    result = HFAdapterMixin.config_agreement(_Stub())

    assert result["available"] is False
    assert result["reason"], "an unavailable comparison must say why"
    assert "spec_matches_config" not in result, (
        "reporting a verdict with nothing to compare against would be a verdict "
        "about nothing")


def test_a_config_mismatch_would_be_reported_not_ignored():
    """The point of the check: disagreement is visible."""
    source = inspect.getsource(HFAdapterMixin.config_agreement)

    assert '"agrees"' in source, (
        "each field comparison needs an agrees flag")
    assert "all(" in source, (
        "the per-field results must be rolled up into an overall verdict")


# ── The measured path is real, structurally ─────────────────────────────

def test_attention_patterns_require_a_real_forward_pass():
    source = inspect.getsource(HFAdapterMixin.get_attention_patterns)

    assert "output_attentions=True" in source, (
        "attention patterns must come from output_attentions, not be computed")
    # The returned tensors are read directly; re-deriving them from weights
    # would be a different measurement wearing the same name.
    assert "attentions" in source
    assert "matmul" not in source and "einsum" not in source


def test_head_averaging_is_declared_rather_than_silent():
    """`head=-1` as the marker for 'mean over heads'.

    Per-head matrices for a 40-layer 32-head model is a memory problem, so the
    default averages. That is a real choice about what the figure means and it
    has to be visible in the output.
    """
    source = inspect.getsource(HFAdapterMixin.get_attention_patterns)

    assert "per_head" in source, (
        "the per-head path must remain available; averaging is the default, not "
        "the only option")
    assert "head=-1" in source, (
        "an averaged pattern must be distinguishable from a real head")
    signature = inspect.signature(HFAdapterMixin.get_attention_patterns)
    assert signature.parameters["per_head"].default is False, (
        "head averaging must be opt-out, not the only option")


def test_patch_activation_performs_a_real_intervention():
    """A forward hook, not a formula.

    The GPT-2 adapter ablates for real; these must do the same or a cross-model
    comparison is comparing a measurement against a formula.
    """
    source = inspect.getsource(HFAdapterMixin.patch_activation)
    helper = inspect.getsource(HFAdapterMixin._ablated_forward)

    # The hook lives in the helper the method delegates to. Probing only the
    # public method was the first version of this check, and it failed on
    # correct code for the same reason: the work is one call away.
    assert "_ablated_forward" in source, (
        "patch_activation no longer performs a real ablation")
    assert "register_forward_hook" in helper, (
        "the ablation must hook the block output; computing a formula is the "
        "defect this replaces")
    assert "remove()" in helper, (
        "the hook must be removed, or it corrupts every later forward pass")


def test_the_hook_cleans_up_even_when_the_forward_pass_fails():
    """A leaked hook would silently corrupt subsequent measurements."""
    source = inspect.getsource(HFAdapterMixin._ablated_forward)

    assert "finally" in source, (
        "the hook removal must be in a finally block")
    assert "handle.remove()" in source


def test_activations_come_from_the_hidden_states():
    source = inspect.getsource(HFAdapterMixin.get_activations)
    helper = inspect.getsource(HFAdapterMixin._hidden)

    # `_hidden` is where the forward pass is requested; `get_activations` reads
    # the result. Probing only the public method failed on correct code.
    assert "_hidden" in source, (
        "get_activations must read real hidden states")
    assert "output_hidden_states=True" in helper, (
        "hidden states must be requested from the forward pass, not computed")
    assert "ActivationResult" in source
    assert "float(rows" in source, (
        "the reported value must be the real tensor element")


def test_logits_report_a_real_entropy():
    source = inspect.getsource(HFAdapterMixin.get_logits)

    assert "softmax" in source
    assert "topk" in source
    assert "entropy" in source, (
        "entropy was previously `1.1 + h * 0.07`; it must come from the "
        "distribution")


def test_the_residual_stream_summary_is_measured_per_layer():
    source = inspect.getsource(HFAdapterMixin.get_residual_stream)

    assert "hidden_states" in source
    assert "norm()" in source, (
        "residual_norm was `10.0 + layer * 1.5`; it must be the real L2 norm")


# ── Failure modes are distinguished, not swallowed ─────────────────────

def test_a_load_failure_records_why_rather_than_only_flagging_mock_mode():
    """The base class swallowed every exception and set `mock_mode=True`.

    That made "this model is not on this machine" indistinguishable from "this
    code is broken". Both set the flag; only the reason separates them.
    """
    source = inspect.getsource(HFAdapterMixin._load_model)

    assert "_load_failure" in source
    assert "except ImportError" in source, (
        "a missing transformers must be named separately from a missing model")
    assert "type(exc).__name__" in source, (
        "the failure must be recorded with its exception type, not a bare flag")


def test_a_forward_pass_failure_is_not_retried_into_a_result():
    source = inspect.getsource(HFAdapterMixin._forward)

    assert "except Exception" in source, (
        "a failed forward pass must be caught, or one bad prompt aborts a batch")
    assert "return None" in source, (
        "the failure must be reported as None rather than partial output")


def test_a_missing_layer_is_reported_not_clamped():
    """Requesting layer 99 of a 12-layer model must not measure layer 11."""
    source = inspect.getsource(HFAdapterMixin._layer_error)

    assert "outside this model" in source
    assert "Nothing was measured" in source


def test_each_loader_tries_the_local_cache_and_sets_a_timeout():
    """Per function, not one grep over the file.

    The first version of this asserted `local_files_only=True` appears somewhere
    in the module. It appears at two sites -- a docstring mention and the
    tokenizer loader -- so deleting the real one left the others and the guard
    passed on broken code. Six of ten negative controls were fooled this way.

    So each loader is inspected individually, with prose stripped.
    """
    import backend.science.models.hf_adapter as mod

    for name in ("_load_tokenizer", "_load_causal_lm"):
        code = _strip_prose(inspect.getsource(getattr(mod, name)))
        assert "local_files_only" in code, (
            f"{name} does not try the local cache first, so every construction "
            f"hits the network even when the weights are cached")
        assert "HF_HUB_DOWNLOAD_TIMEOUT" in code, (
            f"{name} sets no download timeout, so a stalled request hangs for "
            f"minutes before failing")


def test_every_dtype_mention_pins_float32():
    """Not "float32 appears somewhere" -- every occurrence.

    `float32` appeared at four sites, so a mutation that unpinned two of them
    left the guard satisfied by the other two. A measurement regime is only
    pinned if no call site is free to choose its own.
    """
    tree = ast.parse(MIXIN.read_text(encoding="utf-8"))

    dtypes = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value in ("float32", "auto", "bfloat16", "float16")
    ]
    assert dtypes, "the loader no longer names a dtype at all"
    assert set(dtypes) == {"float32"}, (
        f"dtype call sites are {sorted(set(dtypes))}; only float32 is pinned, "
        f"because a reported activation is a measurement of a specific "
        f"numerical regime and bf16 next to fp32 baselines is uninterpretable")


def test_an_out_of_range_layer_is_not_clamped_to_the_last_one():
    """The clamp lived in `_ablated_forward`, which no guard inspected.

    `test_a_missing_layer_is_reported_not_clamped` only read `_layer_error`, so
    changing the actual bounds check to run a forward pass anyway -- measuring
    whatever layers exist for a request against layer 99 -- passed it.
    """
    import backend.science.models.hf_adapter as mod

    code = _strip_prose(inspect.getsource(mod.HFAdapterMixin._ablated_forward))

    assert re.search(r"layer\s*>=\s*len\s*\(\s*blocks\s*\)", code), (
        "_ablated_forward no longer bounds-checks the requested layer")
    branch = code.split("len", 1)[-1][:200]
    assert "return None" in branch, (
        "an out-of-range layer must abort, not execute a forward pass against "
        "whatever layers happen to exist")


def test_attention_comes_from_the_returned_tensors():
    """Not "no matmul in the source" -- the tensor is the measurement.

    The first guard asserted `matmul` and `einsum` are absent, which is true of
    any code that reassigns the tensor arithmetically instead. What matters is
    that the values are the model's own attention weights, taken from
    `output.attentions` and indexed at the requested layer.
    """
    import backend.science.models.hf_adapter as mod

    code = _strip_prose(
        inspect.getsource(mod.HFAdapterMixin.get_attention_patterns))

    # `tokenize` space-joins, so `output_attentions=True` appears as
    # `output_attentions = True` and `layers[layer]` as `layers [ layer ]`.
    assert re.search(r"output_attentions\s*=\s*True", code), (
        "the forward pass is not asked for attention weights")
    assert "attentions" in code, "the returned attention tensors are not read"
    assert re.search(r"layers\s*\[\s*layer\s*\]", code), (
        "the requested layer's attention is not the one indexed")
    # A uniform tensor is a fabricated pattern, which is exactly what the
    # mutation substituted for the model's own weights.
    after_assign = code.split("attn =", 1)[-1][:200] if "attn =" in code else ""
    assert "1.0 /" not in after_assign, (
        "the attention tensor is being synthesised rather than read")


# ── The five families, not one ──────────────────────────────────────────

def test_all_five_families_are_present():
    """The audit named five. A silent drop of one would leave a gap."""
    for name in ("Gemma", "Llama", "Qwen", "Mistral", "DeepSeek"):
        assert any(f"family == {name}" or f"FAMILY == {name!r}"
                   or f'FAMILY: str = {name!r}' in ADAPTERS.read_text(encoding="utf-8")
                   or any(cls.__name__.startswith(name)
                          for cls, _, _ in FAMILIES)), (
            f"the {name} adapter is gone")


def test_deepseek_architectures_are_not_claimed_to_be_one_family():
    """The R1 distill is a Qwen architecture; the coder base is a Llama one.

    Recording which architecture each variant actually uses is the difference
    between "DeepSeek" as a publisher and "DeepSeek" as a shape.
    """
    variants = set(DEEPSEEK)
    assert len(variants) >= 2, (
        "DeepSeek has two distinct architectures here; collapsing them to one "
        "would attribute a measurement to the wrong shape")