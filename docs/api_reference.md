# API Reference

The platform exposes a REST API via `backend/api/dispatcher.py`.

## Authentication

All control-plane routes require a bearer token
(`backend/core/auth.py`): `Authorization: Bearer <token>`.
Token source: `MECH_API_TOKEN` env var, else the random token persisted to
`backend/storage/.mech_api_token` (0600) at first startup. Public without
auth: `GET /`, `GET /health`, CORS preflight (`OPTIONS`), and the built
frontend's static files. Everything else — `/api/*`, `/api/v1/*`, `/mcp`,
`/health/subsystems`, `/openapi.json`, `/docs`, `/redoc` — returns
`401 {"detail": "missing/invalid credentials"}` without a valid token.
Loopback binding and Origin are never authorization.

## Request limits

- Request bodies are capped globally at 2 MB (`MECH_MAX_BODY_BYTES`,
  clamped to 1 KB–64 MB). An oversized `Content-Length` or streamed body
  returns `413` before the body is read.
- `POST /api/society/run` admits at most 2 active runs with 5 queued
  (`503`-shaped `{"status": "busy", ...}`); goal is capped at 4096 chars,
  `model_name` at 128.

## Provenance and attestation

Responses carry `provenance` and, on measurement records, `attested`.
`attested: true` is set only by the layer that performed the measurement
(`backend/core/provenance.py`); wrappers pass it through and never assert it.
A record labelled `provenance: "live"` without `attested: true` is not
scientific evidence: `evidence_policy` requires both, plus
`validation_eligible` and `publication_eligible`. `GET /api/models/{name}`
returns the loaded model's real dimensions, or `status: "unavailable"` with
none — and if the requested name differs from the loaded weights it reports
`model_mismatch: true` with `model_requested`/`model_loaded` rather than
answering with another model's shape.

## Core Endpoints
- `GET /api/v1/research_catalog`: Fetches the unified registry of entities.
- `POST /api/v2/research/run_benchmark`: Triggers the BenchmarkRunner for a specified paper.
- `GET /api/v2/reports/{report_id}`: Fetches a generated ReproducibilityReport.
- `POST /api/v2/project/validate`: Validates a project's publication readiness.

## Research Society (Autonomous AI)
- `POST /platform/autonomous/agent_run`: Submits a research goal to the ResearchSociety orchestrator for automated investigation.
