from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Boolean,
    Float,
    DateTime,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Application(Base):
    __tablename__ = "applications"

    id = Column(String, primary_key=True, default=lambda: f"app_{uuid.uuid4().hex[:12]}")
    user_id = Column(String, nullable=False, index=True)
    job_title = Column(String, nullable=False)
    company = Column(String, nullable=False)
    job_description = Column(Text, nullable=True)
    job_url = Column(String, nullable=True)
    job_type = Column(String, nullable=True)  # full-time, part-time, internship, contract
    status = Column(String, default="applied")  # applied, shortlisted, interview, offer, rejected, withdrawn
    resume_id = Column(String, ForeignKey("resumes.id"), nullable=True)
    workflow_id = Column(String, nullable=True)
    workflow_config_id = Column(String, ForeignKey("workflow_configs.id"), nullable=True)
    applied_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    extra_metadata = Column(JSON, default=dict)

    resume = relationship("Resume", back_populates="applications")
    workflow_config = relationship("WorkflowConfig", back_populates="applications")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "job_title": self.job_title,
            "company": self.company,
            "job_description": self.job_description,
            "job_url": self.job_url,
            "job_type": self.job_type,
            "status": self.status,
            "resume_id": self.resume_id,
            "workflow_id": self.workflow_id,
            "workflow_config_id": self.workflow_config_id,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.extra_metadata or {},
        }


class Resume(Base):
    __tablename__ = "resumes"

    id = Column(String, primary_key=True, default=lambda: f"res_{uuid.uuid4().hex[:12]}")
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)  # plain text or markdown
    version = Column(Integer, default=1)
    file_path = Column(String, nullable=True)  # path to PDF/DOCX
    is_primary = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    extra_metadata = Column(JSON, default=dict)

    applications = relationship("Application", back_populates="resume")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "content": self.content,
            "version": self.version,
            "file_path": self.file_path,
            "is_primary": self.is_primary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.extra_metadata or {},
        }


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(String, primary_key=True, default=lambda: f"jd_{uuid.uuid4().hex[:12]}")
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    company = Column(String, nullable=True)
    description = Column(Text, nullable=False)
    requirements = Column(Text, nullable=True)
    responsibilities = Column(Text, nullable=True)
    qualifications = Column(Text, nullable=True)
    job_url = Column(String, nullable=True)
    parsed_skills = Column(JSON, default=list)
    parsed_requirements = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "company": self.company,
            "description": self.description,
            "requirements": self.requirements,
            "responsibilities": self.responsibilities,
            "qualifications": self.qualifications,
            "job_url": self.job_url,
            "parsed_skills": self.parsed_skills or [],
            "parsed_requirements": self.parsed_requirements or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class WorkflowConfig(Base):
    __tablename__ = "workflow_configs"

    id = Column(String, primary_key=True, default=lambda: f"wf_{uuid.uuid4().hex[:12]}")
    user_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    provider = Column(String, default="n8n")  # n8n, qai, make, zapier, custom
    webhook_url = Column(String, nullable=False)
    api_key = Column(String, nullable=True)
    trigger_event = Column(String, default="new_application")
    is_active = Column(Boolean, default=True)
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    applications = relationship("Application", back_populates="workflow_config")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "provider": self.provider,
            "webhook_url": self.webhook_url,
            "api_key": self.api_key,
            "trigger_event": self.trigger_event,
            "is_active": self.is_active,
            "settings": self.settings or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class WhatsappSetting(Base):
    __tablename__ = "whatsapp_settings"

    id = Column(String, primary_key=True, default=lambda: f"ws_{uuid.uuid4().hex[:12]}")
    user_id = Column(String, nullable=False, index=True)
    phone_number = Column(String, nullable=False)
    api_provider = Column(String, default="twilio")  # twilio, whatsapp-business, meta
    api_key = Column(String, nullable=True)
    api_secret = Column(String, nullable=True)
    is_enabled = Column(Boolean, default=True)
    notify_on = Column(JSON, default=lambda: ["applied", "shortlisted", "interview", "offer", "rejected"])
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "phone_number": self.phone_number,
            "api_provider": self.api_provider,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "is_enabled": self.is_enabled,
            "notify_on": self.notify_on or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
