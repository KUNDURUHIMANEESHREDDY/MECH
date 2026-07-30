"""SQLAlchemy Database Configuration and Models.

Supports SQLite for desktop usage and PostgreSQL for server deployments.
"""

from typing import Any
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Default to SQLite for the desktop version.
DATABASE_URL = "sqlite:///./interp_research.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ExperimentRecord(Base):
    """Stores metadata and provenance for a single experiment run."""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(String, unique=True, index=True)
    model_id = Column(String)
    dataset_id = Column(String)
    status = Column(String)
    metrics = Column(JSON)
    provenance = Column(JSON)

class SessionRecord(Base):
    """Immutable ledger of an experiment execution."""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    project_id = Column(String, index=True)
    verification_hash = Column(String, unique=True)
    session_data = Column(JSON)

class ReportRecord(Base):
    """Persisted reproducibility report."""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True)
    paper_id = Column(String)
    pipeline_name = Column(String)
    fidelity_tier = Column(String)
    report_data = Column(JSON)

def init_db() -> None:
    """Initialize the database schema."""
    Base.metadata.create_all(bind=engine)
