"""n8n workflow integration client."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("MECH.jobs.n8n")


class N8nClient:
    """Client for interacting with n8n Webhook API."""

    def __init__(self, webhook_url: str, api_key: Optional[str] = None):
        self.webhook_url = webhook_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-N8N-API-KEY"] = self.api_key
        return headers

    async def trigger_workflow(
        self,
        trigger: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Trigger an n8n workflow via webhook."""
        url = f"{self.webhook_url}/{trigger}"
        try:
            resp = await self._client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            return {"status": "triggered", "workflow_url": url, "response": resp.json()}
        except httpx.HTTPError as e:
            logger.error(f"n8n webhook error: {e}")
            return {"status": "error", "error": str(e), "workflow_url": url}

    async def test_connection(self) -> dict[str, Any]:
        """Test if the n8n webhook endpoint is reachable."""
        try:
            resp = await self._client.get(
                f"{self.webhook_url}/health",
                headers=self._headers(),
            )
            return {"status": "ok", "reachable": resp.status_code < 500}
        except httpx.HTTPError as e:
            return {"status": "error", "reachable": False, "error": str(e)}


class QAIWorkflowClient:
    """Client for QAI (or similar) agentic workflow platforms."""

    def __init__(self, api_base: str, api_key: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def run_agent(
        self,
        agent_name: str,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run an agent workflow."""
        url = f"{self.api_base}/agents/{agent_name}/run"
        try:
            resp = await self._client.post(url, json=input_data, headers=self._headers())
            resp.raise_for_status()
            return {"status": "completed", "agent": agent_name, "result": resp.json()}
        except httpx.HTTPError as e:
            logger.error(f"QAI agent error: {e}")
            return {"status": "error", "error": str(e), "agent": agent_name}


async def trigger_automation_workflow(
    provider: str,
    config: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch to the correct workflow provider."""
    if provider == "n8n":
        client = N8nClient(
            webhook_url=config.get("webhook_url", ""),
            api_key=config.get("api_key"),
        )
        try:
            result = await client.trigger_workflow(
                trigger=config.get("trigger_event", "new_application"),
                payload=payload,
            )
        finally:
            await client.close()
        return result
    elif provider == "qai":
        client = QAIWorkflowClient(
            api_base=config.get("api_base", ""),
            api_key=config.get("api_key", ""),
        )
        try:
            result = await client.run_agent(
                agent_name=config.get("agent_name", "job-applier"),
                input_data=payload,
            )
        finally:
            await client.close()
        return result
    else:
        # Generic webhook for custom providers
        client = httpx.AsyncClient(timeout=30.0)
        try:
            resp = await client.post(
                config.get("webhook_url", ""),
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return {"status": "triggered", "provider": provider}
        except httpx.HTTPError as e:
            return {"status": "error", "error": str(e)}
        finally:
            await client.aclose()
