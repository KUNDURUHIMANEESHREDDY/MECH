# ADR-0003: FastAPI as Backend Framework

**Date**: 2026-08-10
**Status**: accepted
**Deciders**: Architecture Team

## Context

The MECH platform needs a Python backend framework for:
- REST API serving model inference, interpretability endpoints
- WebSocket support for real-time visualization updates
- Async support for long-running benchmark jobs
- OpenAPI/Swagger documentation generation
- Type-safe request/response validation

## Decision

Use **FastAPI** as the primary backend framework with:
- **Pydantic v2** for request/response validation
- **Uvicorn** as ASGI server
- **SQLAlchemy 2.0** + **Alembic** for database
- **Dependency injection** for services and database sessions

## Alternatives Considered

### Alternative 1: Flask
- **Pros**: Simple, mature, large ecosystem
- **Cons**: No native async, no built-in validation, manual OpenAPI
- **Why not**: MECH needs async for model inference and benchmarks

### Alternative 2: Django + DRF
- **Pros**: Batteries included, admin, auth, migrations
- **Cons**: Heavy, synchronous by default, ORM coupling
- **Why not**: Overkill for API-focused service

### Alternative 3: Starlette (raw)
- **Pros**: Minimal, full control
- **Cons**: Reinventing validation, dependency injection, docs
- **Why not**: FastAPI provides these with negligible overhead

## Consequences

### Positive
- Native async/await for model inference and distributed jobs
- Automatic OpenAPI docs at `/docs` and `/redoc`
- Pydantic validation catches errors at boundary
- Dependency injection enables testing and modularity
- Excellent performance (comparable to Node/Go)

### Negative
- Pydantic v2 migration required code changes
- Less built-in than Django (auth, admin, forms)
- **Mitigation**: Custom auth middleware (ADR-0006)

### Risks
- FastAPI breaking changes between minor versions
- **Mitigation**: Pin version, test upgrades in CI