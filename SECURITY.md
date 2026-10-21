# Security policy

MECH is a local-first mechanistic interpretability research tool. It runs as a
desktop application and a loopback HTTP API on the operator's machine. See
[docs/security.md](docs/security.md) for the implemented trust boundaries and
their limits.

## Supported versions

| Version | Supported |
| --- | --- |
| `main` / latest `0.x` release | Yes |
| Older `0.x` releases | No — fixes land on `main` only |

MECH is pre-1.0 (`pyproject.toml` declares `version = "0.1.0"`). There are no
maintained release branches and no backports. If you are running an older
checkout, reproduce against current `main` before reporting.

## Reporting a vulnerability

**Do not open a public issue for a security problem.**

Use GitHub's private security advisory flow on the repository:

<https://github.com/KUNDURUHIMANEESHREDDY/MECH/security/advisories/new>

That is the only reporting channel. There is no security email address: none is
published, and one invented here would be worse than none. If the advisory form
is unavailable to you, open a normal issue that says only *"I have a security
report and need a private channel"* — with no technical detail — and a
maintainer will follow up.

Please include:

- the affected commit or version (`git rev-parse HEAD`),
- the component — backend route, Electron IPC/preload, plugin install/sandbox,
  path containment, or storage,
- reproduction steps, and whether a real API token or a malicious local file is
  required,
- what an attacker gains, in terms of the trust boundaries in
  [docs/security.md](docs/security.md).

## What to expect

| Stage | Expectation |
| --- | --- |
| Acknowledgement | Within 7 days |
| Initial assessment (in scope / out of scope, severity) | Within 14 days |
| Fix or documented decision | Depends on severity; critical issues are handled before other work |
| Public disclosure | Coordinated. We will agree a date with you; default is after a fix is on `main`, or 90 days after the report, whichever comes first |

Credit is given in the advisory and the release notes unless you ask otherwise.

## In scope

- Authentication bypass, token disclosure, or any path to the control plane that
  does not present a valid bearer token.
- Origin/CORS handling that lets a non-allowlisted web origin drive the control
  plane.
- Plugin install, sandbox, or path-containment escapes — reading or executing
  something outside `backend/plugins/installed/` or an allowed workspace root.
- Path traversal in dataset preview, artifact content, SAE checkpoint loading, or
  MCP workspace tools.
- Unsafe deserialization of checkpoints, manifests or request bodies.
- Electron IPC/preload weaknesses that give the renderer more than the fixed
  `appApi` surface, or that expose the token beyond the documented renderer
  access.
- Result fabrication reachable *through* a request: a code path that turns a
  failure into a success payload with plausible data.

## Out of scope

- Anything requiring the operator's own environment to be already compromised
  (same-user process reading `backend/storage/.mech_api_token`, malware on the
  machine, a hostile `MECH_*` environment variable set before launch).
- Deliberate exposure via `MECH_BIND_HOST` without TLS. Binding wider is a
  documented operator choice; the cleartext consequence is stated in
  [docs/security.md](docs/security.md) §2.
- Renderer-side XSS treated as "token theft" on its own — the token is
  intentionally available to renderer script (`frontend/electron/preload.js`).
  Report the injection vector instead.
- Missing hardening that is documented as absent: no rate limiting, no per-user
  authorization, no audit identity, no network isolation for plugin workers.
  These are known limits listed in [docs/security.md](docs/security.md) §10, not
  vulnerabilities. A platform that silently refuses a plugin worker resource
  limit is worth reporting, because that is a control the code claims to apply.
- Vulnerabilities in `torch`, `transformers`, `transformer_lens`,
  `huggingface_hub`, Electron or other third-party dependencies. Report those
  upstream; tell us only if MECH uses them in a way that widens the impact.
- Model-output quality, benchmark correctness, or scientific interpretation.
  Those are correctness issues — open a normal issue.

## For maintainers

- Never convert an exception into a successful response to make a report go
  away. `AGENTS.md` forbids it and the fix will be rejected.
- A fix for a containment or auth issue needs a regression test that fails
  before the fix, exercising the real rejection path.
- Add an advisory entry and a `CHANGELOG.md` "Security" line under `Unreleased`.
