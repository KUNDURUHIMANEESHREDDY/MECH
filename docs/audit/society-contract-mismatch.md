# Society route: pre-existing backend contract mismatch

**Status:** confirmed, reproduced, root-caused. Fixed separately from the
`scripts/` audit work so the two sets of test results stay distinguishable.
**Found by:** `scripts/capture_screenshots.cjs` (the assertion-based capture
rewrite), which failed the `society` route instead of screenshotting the error.

## The failure

```
ResearchSocietyV2().run("Reproduce IOI on gpt2-small")
TypeError: Executor.reproduce() got an unexpected keyword argument 'model_name'
```

Reproduced in isolation, with no browser and no capture script:

```
PYTHONPATH=. python -c "import asyncio; from backend.agents.society import ResearchSocietyV2; asyncio.run(ResearchSocietyV2().run('...'))"
```

The UI surfaces it as `Society run failed — Executor.reproduce() got an
unexpected keyword argument 'model_name'`, and the capture script exits 1.

## Root cause — NOT a `model_name` → `model_variant` mapping problem

The initial report described this as `planner.py` emitting `model_name="gpt2"`
while the pipeline expects `model_variant`. That framing is wrong, and acting
on it would have produced a plausible-looking but incorrect fix. The traced
contract:

`backend/agents/planner.py:176` emits the workflow's **anchor** node:

```python
offer({"id": "load", "agent": "executor", "op": "reproduce",
       "args": {"model_name": "gpt2"}, "state": "Experiment",
       "rationale": "Every downstream step needs live weights."},
      "unconditional: no measurement is possible without weights")
```

Three pieces of evidence identify the defect as **the wrong `op`**:

1. **The node's own rationale says what it intends.** `"id": "load"` and
   *"Every downstream step needs live weights"* describe loading a model, not
   reproducing a paper.
2. **`Executor.ensure_model(model_name)` is the method that takes
   `model_name`**, and it does precisely that: `await self._call("load")`, then
   `res["model_name"] = model_name`. It is dispatchable on the supervisor
   (`self.executor = Executor()`, and `_agent_methods()` confirms
   `ensure_model` is exposed).
3. **`model_variant` is not reachable from this call site at all.**
   `reproduce(paper_id, n_prompts)` does not forward to it. `model_variant`
   (`"small"` / `"medium"`, composed as `f"gpt2-{model_variant}"` in
   `GPT2Adapter`) is a *pipeline-run* parameter on `IOIPipeline.run`, set by the
   pipeline itself — a different concept from a loaded model identity
   (`SUPPORTED_MODEL = "gpt2"`, `engine.load()`). Mapping `model_name` onto it
   would conflate two unrelated identifiers.

So `op` should be `ensure_model`. The node is a copy-paste slip: the `op` field
was carried over from the `reproduce` node that follows it at line 185, while
the `id`, `rationale` and `args` all describe a model load.

## Why the existing guards did not catch it

`Planner.offer()` validates that the named `op` **exists** on the agent
(`if methods and op not in methods`). `reproduce` does exist, so the node
passed. Nothing validated that the node's `args` were *accepted by* that `op`.
The `load` node is therefore a plan that is guaranteed to fail at execution —
the guard checked dispatchability, not the call signature.

## Blast radius

- The `society` route fails for **every** goal, not only IOI: the anchor node is
  unconditional.
- Committed `docs/images/ui/06-research-society.png` and
  `06b-research-society-progress.png` were **byte-identical**
  (`sha256 fd42ac6652641cb6289ab91a`, 121170 bytes). The old capture script
  screenshotted the same failed page as both "started" and "progress".
- No other caller depends on `op="reproduce"` for the anchor node; nothing in
  `backend/agents/`, `backend/api/` or `tests/` reads it.

## Fix

One line in `backend/agents/planner.py`: `"op": "reproduce"` → `"op":
"ensure_model"` on the node whose `id` is `"load"`. No signature changes, no
alias, no argument dropped.

Deliberately **not** done:

- **No compatibility alias** on `reproduce` accepting `model_name`. There is no
  evidence any caller needs one, and an alias would hide the next mismatch.
- **No change to `reproduce`'s signature.** It is correct as written; the caller
  was wrong.
- **No change to `model_variant`.** Different concept, different call site.

## Regression tests added

`tests/pytest/test_society_node_contract.py`:

- every node the planner emits binds against its target method's signature
  (the missing guard — catches any future copy-paste of this shape);
- the anchor node dispatches `ensure_model`, not `reproduce`;
- the anchor's `model_name` reaches `ensure_model` intact;
- the full plan is executable: each node's args bind;
- capture exit codes: genuine failure → nonzero, success → 0.

## Result after the fix

The anchor node runs, and the workflow proceeds through four further steps:

```
load       executor    loaded      <- was TypeError for every goal
reproduce  executor    completed
inspect    inspector   ok
patch      executor    ok
discover   discoverer  unavailable  <- a different, deliberate gate (below)
```

The Society route no longer dies at its first step. The capture script now
reaches the trace panel and records five steps where it previously recorded a
`TypeError`.

### The Society route still does not reach `phase="done"`

It ends in `phase="failed"` at the **discover** stage:

```
Discovery blocked: stage status 'unavailable' is not complete.
```

This is **not** the contract mismatch and **not** a regression from this fix. It
is a separate, deliberate statistical gate:

- `backend/interpretability/discovery/live_discovery.py` sets
  `MIN_PROMPTS_FOR_VALIDATION = 10`, and its `run()` defaults to `n_prompts=4`
  and `n_interaction_prompts=2` (`MIN_PROMPTS_FOR_INTERACTION = 5`).
- `DiscoveryEngine.discover_and_orchestrate()` takes **only**
  `hypothesis_statement` — it does not forward a prompt count.
- So the engine always runs 4 prompts, and `evidence_policy.discovery_is_live()`
  requires `validation_eligible is True and publication_eligible is True`,
  which 4 prompts can never satisfy.
- The engine reports this itself: `ineligible_because: ["4 prompts, below the 10
  needed for a validation decision", "pairwise edges were averaged from 2
  prompts, below the 5 needed to support a claim about them"]`.

The `live_discovery.py` docstring states the intent explicitly: 2 prompts are
"enough to compute a mean and not enough to support a claim about it", and the
default was made a parameter rather than raised "so the cost decision stays with
the caller".

**Why the threshold is left alone:** raising it to make the route green would
mean running a 10-prompt live discovery on every Society request. That is a cost
decision, and `live_discovery.py` documents having deliberately left it with the
caller. Threading `n_prompts` through `discover_and_orchestrate` →
`LiveIOIDiscovery.run` would be the coherent route, but it touches four files
this session did not author and changes runtime cost — outside a fix whose scope
is one argument name.

Not fixed; recorded here so it is not confused with the mismatch.

### The capture script's Society verdict was wrong at first

With only the planner fix, the route **passed and exited 0** — over a workflow
that had reported `Society run failed`. The trace panel and the error banner
render independently, so "steps appeared" and "the run failed" are both true at
once, and waiting for steps alone reported success.

The verdict is now the run's **terminal phase** (`data-phase` on
`[data-testid="society-status"]`), not the presence of a trace. `done` is the only
pass; `failed`, `stopped`, and "never settled" all fail with the error text. The
route now exits `1` with:

```
society: the Society run ended in phase "failed", not "done":
Society request failed / Discovery blocked: stage status 'unavailable' is not complete.
```

## Remaining uncertainty

- `society.py` and `executor.py` both carry unrelated uncommitted edits
  (`attested` propagation). Those were preserved byte-for-byte; the fix touches
  only `planner.py`, which is unmodified in the working tree.
- `plan()` has a `MAX_STEPS` truncation. The current plan emits 7 nodes, so
  nothing is truncated today. Not addressed — out of scope, and no evidence it
  is a live problem.