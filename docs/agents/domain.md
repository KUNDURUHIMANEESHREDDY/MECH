# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root, or
- **`GLOSSARY-MAP.md`** at the repo root if it exists: it points at one `GLOSSARY.md` per context. Read each one relevant to the topic.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in. In multi-context repos, also check `src/<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo (most repos):

```
/
├── GLOSSARY.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

Multi-context repo (presence of `GLOSSARY-MAP.md` at the root):

```
/
├── GLOSSARY-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── GLOSSARY.md
    │   └── docs/adr/                  ← context-specific decisions
    └── billing/
        ├── GLOSSARY.md
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_

---

## This repo: single-context

**Layout: single-context.** One `GLOSSARY.md` and one `docs/adr/` at the repo
root, both not yet created. There is no `GLOSSARY-MAP.md` and no `packages/`.

MECH has two layers that read like separate contexts but share one vocabulary
and one set of invariants, so splitting them into per-context glossaries would
force two files to define `evidence level`, `provenance`, and `NOT_EXECUTED`
twice — and those are precisely the terms where drift would be dangerous. Keep
them in one root glossary, and group entries by layer if it grows unwieldy.

### Terms already load-bearing across layers

These appear in both the runtime and the interpretability engines, and
substituting a synonym for them is the failure mode the glossary exists to
prevent:

| Term | Means | Not to be confused with |
| ---- | ----- | ----------------------- |
| **evidence level** | The graded strength of support a result carries. | A confidence score, or a probability |
| **provenance** | The record of which model, revision, prompt, config and seed produced a result. | Logging, or a timestamp |
| **`NOT_EXECUTED`** | The terminal status for a registered engine that did not run. Distinct from `FAILED`. | An empty successful result |
| **seam** | The public boundary a test observes behaviour through. | An internal helper |

If a new term is needed that is not in the glossary yet, note it rather than
coining it ad hoc.

### Where the scientific vocabulary currently lives

In the absence of a glossary, the existing prose docs are the closest thing to
one, and they are worth reading before touching an engine:

- `docs/scientific_methods.md` — measurement and causal-claim vocabulary
- `docs/SCIENTIFIC_VALIDATION.md` — evidence grading and validation gates
- `docs/reproducibility.md` — seeds, tolerances, provenance requirements
- `docs/architecture.md` — layer boundaries

Terminology in `backend/validation/` and `backend/interpretability/` will
usually match these. Where it does not, that mismatch is a finding worth
filing, not a definition to adopt.