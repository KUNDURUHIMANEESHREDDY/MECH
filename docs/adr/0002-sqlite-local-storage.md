# ADR-0002: SQLite for Local Development Storage

**Date**: 2026-08-10
**Status**: accepted
**Deciders**: Architecture Team

## Context

The MECH platform needs local persistence for:
- Experiment metadata and results
- User sessions and workspace state
- Plugin configurations
- Cached model outputs

During development, we need zero-config, zero-infrastructure persistence.

## Decision

Use **SQLite** (via `sqlite3`/`aiosqlite`) for local development storage with `DesktopStorage` abstraction layer.

Production deployments will use PostgreSQL via the same repository interface.

## Alternatives Considered

### Alternative 1: PostgreSQL for Development
- **Pros**: Production parity, catches SQL dialect issues early
- **Cons**: Requires Docker/service management, not zero-config
- **Why not**: Adds friction for new contributors

### Alternative 2: JSON File Storage
- **Pros**: Zero dependencies, human-readable
- **Cons**: No query capability, no transactions, scales poorly
- **Why not**: Insufficient for experiment metadata queries

### Alternative 3: DuckDB
- **Pros**: Embedded, analytical queries, Parquet support
- **Cons**: Less familiar, different SQL dialect
- **Why not**: Overkill for current needs

## Consequences

### Positive
- Zero-config development setup (`python main.py` works immediately)
- Single file database (`mech.db`) - easy backup/copy/share
- ACID transactions for experiment integrity
- Same repository interface for PostgreSQL in production

### Negative
- SQLite limitations: no concurrent writers, limited ALTER TABLE
- SQL dialect differences with PostgreSQL (mitigated by SQLAlchemy ORM)
- Not suitable for production multi-user loads

### Risks
- Schema drift between SQLite/PostgreSQL
- **Mitigation**: Use SQLAlchemy migrations (Alembic) tested against both
- **Mitigation**: CI tests against PostgreSQL in pipeline