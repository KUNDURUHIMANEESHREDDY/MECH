# Contributing

Thanks for your interest in MECH.

## Guidelines

- Follow PEP 8 for Python code, and add type hints to new public functions.
- Write tests for any new module. **Prefer a test against live weights over another mock.**
  Only `tests/pytest/test_validation_loop.py` currently touches a real model, so the rest of
  the suite cannot tell you the interpretability is correct.
- Make the Python suite pass before opening a PR:
  ```bash
  python -m pytest tests/pytest -q
  ```
  This is currently **186 passed / 27 failed / 2 collection errors** — most failures are
  tests that still assert the old synthetic-success contract while the code correctly fails
  closed. See [What's next](roadmap.md) in the root README.
- Run the frontend suites too:
  ```bash
  cd frontend
  npm run test:js          # currently 8 passed, 1 file fails to load
  npx playwright test      # 4 offline specs, same as CI
  ```
- Document any API changes in `docs/api_reference.md`.

## The provenance rule

This is the one rule that matters most in this codebase.

**If you touch anything that produces a number, preserve its provenance tag.** Every API
response carries `provenance` (`live` / `seeded` / `reference` / `unavailable`), and most
carry a per-field `field_provenance` map.

A result that loses its `live` marker is a regression, even if the tests pass. Conversely, if
an executor genuinely is not implemented, returning `unavailable` with a `reason` is the
correct behaviour — not a placeholder number.

```python
# Good: fails closed, explains itself
if not (engine and engine.is_available()):
    return {"status": "unavailable", "reason": "No live executor is connected", ...}

# Bad: plausible-looking fiction
return {"score": 0.94, "fidelity": 0.97}
```

The evidence graph (`backend/core/evidence_graph.py`) and the publication gate
(`backend/agents/evidence_policy.py`) enforce this in code. Please don't route around them.

## Reporting a bug

Please include:

- What you ran, and the exact output.
- The `provenance` tag of the response you expected to be live.
- Whether GPT-2 weights loaded (`GET /api/gpt2/load` returns `status`).
