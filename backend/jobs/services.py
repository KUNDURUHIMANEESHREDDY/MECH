"""Core job automation service."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from backend.storage.database import DesktopStorage
from backend.jobs.models import Application, Resume, JobDescription, WorkflowConfig, WhatsappSetting

logger = logging.getLogger("MECH.jobs.service")


class JobAutomationService:
    """Main service for job application automation."""

    def __init__(self, storage: DesktopStorage):
        self._storage = storage

    # ------------------------------------------------------------------
    # Applications
    # ------------------------------------------------------------------
    def list_applications(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List applications for a user, optionally filtered by status."""
        apps = self._storage.list_experiments()  # reuse JSON storage pattern
        # Filter by user_id and status
        result = []
        for app_data in apps:
            if isinstance(app_data, dict) and app_data.get("user_id") == user_id:
                if status and app_data.get("status") != status:
                    continue
                result.append(app_data)
        return result[:limit]

    def create_application(
        self,
        user_id: str,
        job_title: str,
        company: str,
        job_description: Optional[str] = None,
        job_url: Optional[str] = None,
        job_type: Optional[str] = None,
        resume_id: Optional[str] = None,
        workflow_config_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new application record."""
        app = Application(
            user_id=user_id,
            job_title=job_title,
            company=company,
            job_description=job_description,
            job_url=job_url,
            job_type=job_type,
            resume_id=resume_id,
            workflow_config_id=workflow_config_id,
        )
        return app.to_dict()

    def update_application_status(
        self,
        application_id: str,
        status: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Update the status of an application."""
        # In production, this would update the database record
        return {
            "status": "updated",
            "application_id": application_id,
            "new_status": status,
            "metadata": metadata or {},
        }

    # ------------------------------------------------------------------
    # Resumes
    # ------------------------------------------------------------------
    def list_resumes(self, user_id: str) -> list[dict[str, Any]]:
        """List all resumes for a user."""
        resumes = self._storage.list_experiments()  # reuse storage
        return [r for r in resumes if isinstance(r, dict) and r.get("user_id") == user_id]

    def get_resume(self, resume_id: str, user_id: str) -> Optional[dict[str, Any]]:
        """Get a specific resume."""
        resumes = self.list_resumes(user_id)
        for r in resumes:
            if r.get("id") == resume_id:
                return r
        return None

    def create_resume(
        self,
        user_id: str,
        name: str,
        content: str,
        file_path: Optional[str] = None,
        is_primary: bool = False,
    ) -> dict[str, Any]:
        """Create a new resume."""
        resume = Resume(
            user_id=user_id,
            name=name,
            content=content,
            file_path=file_path,
            is_primary=is_primary,
        )
        return resume.to_dict()

    def update_resume(self, resume_id: str, user_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a resume."""
        return {"status": "updated", "resume_id": resume_id, "updates": kwargs}

    # ------------------------------------------------------------------
    # Resume Auto-Update
    # ------------------------------------------------------------------
    def auto_update_resume_for_job(
        self,
        resume_id: str,
        user_id: str,
        job_description: str,
    ) -> dict[str, Any]:
        """Auto-update a resume based on a job description."""
        resume = self.get_resume(resume_id, user_id)
        if not resume:
            return {"status": "error", "error": "Resume not found"}

        from backend.jobs.resume_updater import auto_update_resume

        result = auto_update_resume(resume["content"], job_description)

        # Save the updated resume
        updated = self.update_resume(resume_id, user_id, content=result["updated_resume"])

        return {
            "status": "updated",
            "resume_id": resume_id,
            "changes": result["changes_made"],
            "analysis": result["analysis"],
            "graduation_required": result["graduation_required"],
            "is_internship": result["is_internship"],
            "updated_resume_preview": result["updated_resume"][:500],
        }

    # ------------------------------------------------------------------
    # Job Descriptions
    # ------------------------------------------------------------------
    def save_job_description(
        self,
        user_id: str,
        title: str,
        description: str,
        company: Optional[str] = None,
        requirements: Optional[str] = None,
        responsibilities: Optional[str] = None,
        qualifications: Optional[str] = None,
        job_url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Save a job description for later reference."""
        jd = JobDescription(
            user_id=user_id,
            title=title,
            company=company,
            description=description,
            requirements=requirements,
            responsibilities=responsibilities,
            qualifications=qualifications,
            job_url=job_url,
        )
        return jd.to_dict()

    # ------------------------------------------------------------------
    # Workflow Configs
    # ------------------------------------------------------------------
    def list_workflow_configs(self, user_id: str) -> list[dict[str, Any]]:
        """List workflow configurations for a user."""
        configs = self._storage.list_experiments()
        return [c for c in configs if isinstance(c, dict) and c.get("user_id") == user_id]

    def create_workflow_config(
        self,
        user_id: str,
        name: str,
        provider: str,
        webhook_url: str,
        api_key: Optional[str] = None,
        trigger_event: str = "new_application",
        settings: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a new workflow configuration."""
        config = WorkflowConfig(
            user_id=user_id,
            name=name,
            provider=provider,
            webhook_url=webhook_url,
            api_key=api_key,
            trigger_event=trigger_event,
            settings=settings or {},
        )
        return config.to_dict()

    # ------------------------------------------------------------------
    # WhatsApp Settings
    # ------------------------------------------------------------------
    def get_whatsapp_settings(self, user_id: str) -> Optional[dict[str, Any]]:
        """Get WhatsApp settings for a user."""
        settings = self._storage.list_experiments()
        for s in settings:
            if isinstance(s, dict) and s.get("user_id") == user_id and "phone_number" in s:
                return s
        return None

    def save_whatsapp_settings(
        self,
        user_id: str,
        phone_number: str,
        api_provider: str = "twilio",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        notify_on: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Save WhatsApp notification settings."""
        setting = WhatsappSetting(
            user_id=user_id,
            phone_number=phone_number,
            api_provider=api_provider,
            api_key=api_key,
            api_secret=api_secret,
            notify_on=notify_on or ["applied", "shortlisted", "interview", "offer", "rejected"],
        )
        return setting.to_dict()

    # ------------------------------------------------------------------
    # Automation Triggers
    # ------------------------------------------------------------------
    async def trigger_workflow(
        self,
        workflow_config_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Trigger an automation workflow."""
        from backend.jobs.n8n_client import trigger_automation_workflow

        configs = self.list_workflow_configs("")
        config = None
        for c in configs:
            if c.get("id") == workflow_config_id:
                config = c
                break

        if not config:
            return {"status": "error", "error": "Workflow config not found"}

        return await trigger_automation_workflow(
            provider=config.get("provider", "n8n"),
            config=config,
            payload=payload,
        )

    async def apply_to_job(
        self,
        user_id: str,
        job_title: str,
        company: str,
        job_description: str,
        job_url: str,
        resume_id: str,
        workflow_config_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Full automation: update resume, trigger workflow, create application record."""
        # Step 1: Auto-update resume
        resume_update = self.auto_update_resume_for_job(resume_id, user_id, job_description)

        # Step 2: Trigger workflow if configured
        workflow_result = None
        if workflow_config_id:
            workflow_result = await self.trigger_workflow(
                workflow_config_id,
                {
                    "job_title": job_title,
                    "company": company,
                    "job_description": job_description,
                    "job_url": job_url,
                    "resume_id": resume_id,
                    "user_id": user_id,
                },
            )

        # Step 3: Create application record
        application = self.create_application(
            user_id=user_id,
            job_title=job_title,
            company=company,
            job_description=job_description,
            job_url=job_url,
            resume_id=resume_id,
            workflow_config_id=workflow_config_id,
        )

        return {
            "status": "applied",
            "application": application,
            "resume_update": resume_update,
            "workflow_result": workflow_result,
        }
