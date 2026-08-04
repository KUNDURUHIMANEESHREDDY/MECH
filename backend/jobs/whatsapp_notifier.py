"""WhatsApp notification service."""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("MECH.jobs.whatsapp")


class WhatsappNotifier:
    """Send WhatsApp notifications via Twilio or Meta API."""

    def __init__(
        self,
        phone_number: str,
        api_provider: str = "twilio",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.phone_number = phone_number
        self.api_provider = api_provider
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def send_notification(
        self,
        message: str,
        application_id: Optional[str] = None,
        job_title: Optional[str] = None,
        company: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a WhatsApp notification about application status."""
        if self.api_provider == "twilio":
            return await self._send_via_twilio(message)
        elif self.api_provider == "whatsapp-business":
            return await self._send_via_whatsapp_business(message)
        elif self.api_provider == "meta":
            return await self._send_via_meta(message)
        else:
            # Generic webhook fallback
            return await self._send_via_webhook(message)

    async def _send_via_twilio(self, message: str) -> dict[str, Any]:
        """Send via Twilio WhatsApp API."""
        url = "https://api.twilio.com/2010-04-01/Accounts"
        # Twilio WhatsApp uses the Messages API with WhatsApp sender
        try:
            resp = await self._client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{self.api_key}/Messages.json"
                if self.api_key else url,
                data={
                    "To": f"whatsapp:{self.phone_number}",
                    "Body": message,
                    "From": "whatsapp:+14155238886",  # Twilio sandbox number
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                auth=(self.api_key or "", self.api_secret or ""),
            )
            resp.raise_for_status()
            return {"status": "sent", "provider": "twilio", "message_id": resp.json().get("sid")}
        except httpx.HTTPError as e:
            logger.error(f"Twilio WhatsApp error: {e}")
            return {"status": "error", "provider": "twilio", "error": str(e)}

    async def _send_via_whatsapp_business(self, message: str) -> dict[str, Any]:
        """Send via WhatsApp Business API."""
        try:
            resp = await self._client.post(
                "https://graph.facebook.com/v18.0/me/messages",
                json={
                    "messaging_product": "whatsapp",
                    "to": self.phone_number,
                    "type": "text",
                    "text": {"body": message},
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            return {"status": "sent", "provider": "whatsapp-business"}
        except httpx.HTTPError as e:
            logger.error(f"WhatsApp Business error: {e}")
            return {"status": "error", "provider": "whatsapp-business", "error": str(e)}

    async def _send_via_meta(self, message: str) -> dict[str, Any]:
        """Send via Meta WhatsApp API."""
        return await self._send_via_whatsapp_business(message)

    async def _send_via_webhook(self, message: str) -> dict[str, Any]:
        """Send via a generic webhook."""
        try:
            resp = await self._client.post(
                "https://api.whatsapp.com/send",
                json={"phone": self.phone_number, "message": message},
            )
            resp.raise_for_status()
            return {"status": "sent", "provider": "webhook"}
        except httpx.HTTPError as e:
            logger.error(f"WhatsApp webhook error: {e}")
            return {"status": "error", "provider": "webhook", "error": str(e)}

    async def send_application_update(
        self,
        job_title: str,
        company: str,
        status: str,
        message: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send a formatted application status update."""
        if message:
            full_message = message
        else:
            status_emoji = {
                "applied": "📝",
                "shortlisted": "✅",
                "interview": "� interview",
                "offer": "🎉 Offer!",
                "rejected": "❌",
                "withdrawn": "🔄",
            }.get(status, "📋")
            full_message = f"{status_emoji} Application Update\n\nJob: {job_title}\nCompany: {company}\nStatus: {status.upper()}\n\nKeep going!"

        return await self.send_notification(full_message)
