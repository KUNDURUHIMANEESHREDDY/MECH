# API Reference

The platform exposes a REST API via `backend/api/dispatcher.py`.

## Core Endpoints
- `GET /api/v1/research_catalog`: Fetches the unified registry of entities.
- `POST /api/v2/research/run_benchmark`: Triggers the BenchmarkRunner for a specified paper.
- `GET /api/v2/reports/{report_id}`: Fetches a generated ReproducibilityReport.
- `POST /api/v2/project/validate`: Validates a project's publication readiness.

## Research Society (Autonomous AI)
- `POST /platform/autonomous/agent_run`: Submits a research goal to the ResearchSociety orchestrator for automated investigation.
