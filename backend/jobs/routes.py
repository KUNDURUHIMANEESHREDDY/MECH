"""Job automation API routes."""

from fastapi import APIRouter, HTTPException
from typing import Any, Dict, Optional

from backend.utils.url_validator import SSRFViolation, validate_url

router = APIRouter(prefix="/jobs", tags=["job-automation"])


# ---------------------------------------------------------------------------
# Chat / Conversation endpoints
# ---------------------------------------------------------------------------

@router.post("/chat")
async def chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Chat endpoint for job application automation.

    The user describes what they want (job role, internship, etc.)
    and the agent processes the request through the automation pipeline.
    """
    message = payload.get("message", "")
    user_id = payload.get("user_id", "default")
    context = payload.get("context", {})

    # Parse intent from message
    intent = _parse_intent(message)

    result = {
        "status": "ok",
        "intent": intent,
        "message": message,
        "user_id": user_id,
    }

    # Handle different intents
    if intent == "apply":
        result["action"] = "apply"
        result["details"] = _handle_apply(message, context)
    elif intent == "track":
        result["action"] = "track"
        result["details"] = _handle_track(user_id, context)
    elif intent == "resume_update":
        result["action"] = "resume_update"
        result["details"] = _handle_resume_update(message, context)
    elif intent == "auth":
        result["action"] = "auth"
        result["details"] = _handle_auth(message, user_id)
    elif intent == "status_check":
        result["action"] = "status_check"
        result["details"] = _handle_status_check(user_id, context)
    else:
        result["action"] = "clarify"
        result["details"] = {
            "message": "I can help you with job applications! Tell me what job or internship you want to apply for.",
            "supported_actions": ["apply", "track", "resume_update", "auth", "status_check"],
        }

    return result


@router.get("/chat/help")
async def chat_help() -> Dict[str, Any]:
    """Return help text for the chat interface."""
    return {
        "commands": {
            "apply": "Apply for a job or internship - describe the role",
            "track": "Check your application status",
            "resume_update": "Update your resume for a specific job",
            "auth": "Authenticate with a service (n8n, etc.)",
            "status_check": "Check if you got selected for any applications",
        },
        "examples": [
            "I want to apply for a software engineering internship at Google",
            "Apply for a data scientist role at Microsoft",
            "Check my application status",
            "Update my resume for a ML engineer position",
        ],
    }


# ---------------------------------------------------------------------------
# Application CRUD
# ---------------------------------------------------------------------------

@router.get("/applications")
async def list_applications(user_id: str = "", status: Optional[str] = None) -> Dict[str, Any]:
    """List all applications for a user."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    apps = service.list_applications(user_id or "default", status=status)
    return {"applications": apps, "count": len(apps)}


@router.post("/applications")
async def create_application(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new application record."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    app = service.create_application(
        user_id=payload.get("user_id", "default"),
        job_title=payload.get("job_title", ""),
        company=payload.get("company", ""),
        job_description=payload.get("job_description"),
        job_url=payload.get("job_url"),
        job_type=payload.get("job_type"),
        resume_id=payload.get("resume_id"),
        workflow_config_id=payload.get("workflow_config_id"),
    )
    return {"status": "created", "application": app}


@router.patch("/applications/{application_id}/status")
async def update_application_status(
    application_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Update the status of an application."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    result = service.update_application_status(
        application_id,
        payload.get("status", "applied"),
        payload.get("metadata"),
    )
    return result


# ---------------------------------------------------------------------------
# Resume endpoints
# ---------------------------------------------------------------------------

@router.get("/resumes")
async def list_resumes(user_id: str = "") -> Dict[str, Any]:
    """List all resumes for a user."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    resumes = service.list_resumes(user_id or "default")
    return {"resumes": resumes}


@router.post("/resumes")
async def create_resume(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new resume."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    resume = service.create_resume(
        user_id=payload.get("user_id", "default"),
        name=payload.get("name", "My Resume"),
        content=payload.get("content", ""),
        file_path=payload.get("file_path"),
        is_primary=payload.get("is_primary", False),
    )
    return {"status": "created", "resume": resume}


@router.post("/resumes/{resume_id}/update")
async def update_resume_for_job(
    resume_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Auto-update a resume based on a job description."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    result = service.auto_update_resume_for_job(
        resume_id=resume_id,
        user_id=payload.get("user_id", "default"),
        job_description=payload.get("job_description", ""),
    )
    return result


# ---------------------------------------------------------------------------
# Job Description endpoints
# ---------------------------------------------------------------------------

@router.post("/job-descriptions")
async def save_job_description(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Save a job description for later reference."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    jd = service.save_job_description(
        user_id=payload.get("user_id", "default"),
        title=payload.get("title", ""),
        description=payload.get("description", ""),
        company=payload.get("company"),
        requirements=payload.get("requirements"),
        responsibilities=payload.get("responsibilities"),
        qualifications=payload.get("qualifications"),
        job_url=payload.get("job_url"),
    )
    return {"status": "saved", "job_description": jd}


# ---------------------------------------------------------------------------
# Workflow Configuration endpoints
# ---------------------------------------------------------------------------

@router.get("/workflows")
async def list_workflows(user_id: str = "") -> Dict[str, Any]:
    """List workflow configurations."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    workflows = service.list_workflow_configs(user_id or "default")
    return {"workflows": workflows}


@router.post("/workflows")
async def create_workflow(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new workflow configuration."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    config = service.create_workflow_config(
        user_id=payload.get("user_id", "default"),
        name=payload.get("name", ""),
        provider=payload.get("provider", "n8n"),
        webhook_url=validate_url(payload.get("webhook_url", "") or "", allow_private=False, allow_localhost=False),
        trigger_event=payload.get("trigger_event", "new_application"),
        settings=payload.get("settings"),
    )
    return {"status": "created", "workflow": config}


@router.post("/workflows/{workflow_id}/trigger")
async def trigger_workflow(workflow_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger a workflow manually."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    result = await service.trigger_workflow(workflow_id, payload)
    return result


# ---------------------------------------------------------------------------
# WhatsApp Notification endpoints
# ---------------------------------------------------------------------------

@router.get("/whatsapp/settings")
async def get_whatsapp_settings(user_id: str = "") -> Dict[str, Any]:
    """Get WhatsApp notification settings."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    settings = service.get_whatsapp_settings(user_id or "default")
    return {"settings": settings}


@router.post("/whatsapp/settings")
async def save_whatsapp_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Save WhatsApp notification settings."""
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)
    settings = service.save_whatsapp_settings(
        user_id=payload.get("user_id", "default"),
        phone_number=payload.get("phone_number", ""),
        api_provider=payload.get("api_provider", "twilio"),
        api_key=payload.get("api_key"),
        api_secret=payload.get("api_secret"),
        notify_on=payload.get("notify_on"),
    )
    return {"status": "saved", "settings": settings}


@router.post("/whatsapp/send")
async def send_whatsapp_notification(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a WhatsApp notification."""
    from backend.jobs.whatsapp_notifier import WhatsappNotifier

    notifier = WhatsappNotifier(
        phone_number=payload.get("phone_number", ""),
        api_provider=payload.get("api_provider", "twilio"),
        api_key=payload.get("api_key"),
        api_secret=payload.get("api_secret"),
    )
    try:
        result = await notifier.send_notification(
            message=payload.get("message", ""),
            application_id=payload.get("application_id"),
            job_title=payload.get("job_title"),
            company=payload.get("company"),
            status=payload.get("status"),
        )
    finally:
        await notifier.close()
    return result


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/initiate")
async def initiate_auth(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Initiate an authentication flow. Returns a link for the user."""
    from backend.jobs.auth_flow import create_auth_session

    service = payload.get("service", "n8n")
    user_id = payload.get("user_id", "default")
    result = create_auth_session(service, user_id)
    return result


@router.post("/auth/complete")
async def complete_auth(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Complete an authentication flow after the user has visited the link."""
    from backend.jobs.auth_flow import complete_auth_session

    auth_id = payload.get("auth_id", "")
    state = payload.get("state", "")
    result = payload.get("result", {})
    return complete_auth_session(auth_id, state, result)


@router.get("/auth/status/{auth_id}")
async def auth_status(auth_id: str) -> Dict[str, Any]:
    """Check authentication status."""
    from backend.jobs.auth_flow import AuthFlowManager

    manager = AuthFlowManager()
    return manager.get_auth_status(auth_id)


# ---------------------------------------------------------------------------
# Full automation endpoint
# ---------------------------------------------------------------------------

@router.post("/automate")
async def automate_application(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Full automation: update resume, trigger workflow, create application.

    This is the main endpoint that ties everything together.
    """
    from backend.storage.database import DesktopStorage
    from backend.jobs.services import JobAutomationService

    storage = DesktopStorage()
    service = JobAutomationService(storage)

    result = await service.apply_to_job(
        user_id=payload.get("user_id", "default"),
        job_title=payload.get("job_title", ""),
        company=payload.get("company", ""),
        job_description=payload.get("job_description", ""),
        job_url=payload.get("job_url"),
        resume_id=payload.get("resume_id", ""),
        workflow_config_id=payload.get("workflow_config_id"),
    )
    return result


# ---------------------------------------------------------------------------
# Intent parsing helpers
# ---------------------------------------------------------------------------

def _parse_intent(message: str) -> str:
    """Parse user intent from their message."""
    msg_lower = message.lower()

    # Check for apply/intent keywords
    apply_keywords = [
        "apply", "application", "job", "internship", "position", "role",
        "want to work", "looking for", "hiring", "open position",
    ]
    for kw in apply_keywords:
        if kw in msg_lower:
            return "apply"

    # Check for tracking/status keywords
    track_keywords = ["status", "track", "how is", "did i get", "selected", "result"]
    for kw in track_keywords:
        if kw in msg_lower:
            return "track"

    # Check for resume update keywords
    resume_keywords = ["resume", "update resume", "tailor", "modify resume"]
    for kw in resume_keywords:
        if kw in msg_lower:
            return "resume_update"

    # Check for auth keywords
    auth_keywords = ["login", "auth", "sign in", "authenticate", "connect"]
    for kw in auth_keywords:
        if kw in msg_lower:
            return "auth"

    return "clarify"


def _handle_apply(message: str, context: dict) -> dict:
    """Handle a job application request."""
    return {
        "message": "I'll help you apply for this job. Please provide the job title and company name.",
        "step": "collect_job_details",
    }


def _handle_track(user_id: str, context: dict) -> dict:
    """Handle a tracking request."""
    return {
        "message": "Fetching your application status...",
        "step": "fetch_status",
    }


def _handle_resume_update(message: str, context: dict) -> dict:
    """Handle a resume update request."""
    return {
        "message": "I'll update your resume based on the job description. Please share the job details.",
        "step": "collect_job_description",
    }


def _handle_auth(message: str, user_id: str) -> dict:
    """Handle an authentication request."""
    from backend.jobs.auth_flow import create_auth_session

    result = create_auth_session("n8n", user_id)
    return result


def _handle_status_check(user_id: str, context: dict) -> dict:
    """Handle a status check request."""
    return {
        "message": "Checking your application status...",
        "step": "check_status",
    }
