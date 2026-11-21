"""Webhook support for Furrow external integrations.

This module provides functionality to send events to external webhook URLs
for integration with other systems (CI/CD, monitoring, notifications, etc.).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

from furrow.config import Settings
from furrow.core.events import Event, EventType

logger = logging.getLogger(__name__)


class WebhookClient:
    """Client for sending events to a webhook URL."""

    def __init__(
        self,
        url: str,
        secret: str | None = None,
        events: list[str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.url = url
        self.secret = secret
        self.events = events
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _should_send(self, event_type: str) -> bool:
        """Check if this event type should be sent to the webhook."""
        if self.events is None:
            return True
        return event_type in self.events

    def _sign_payload(self, payload: bytes) -> str:
        """Generate HMAC-SHA256 signature for the payload."""
        if not self.secret:
            return ""
        return hmac.new(
            self.secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

    async def send(self, event: Event) -> bool:
        """Send an event to the webhook URL.

        Returns True if the webhook was sent successfully (2xx response),
        False otherwise.
        """
        if not self._should_send(event.type.value if isinstance(event.type, EventType) else str(event.type)):
            return True

        client = await self._get_client()

        # Prepare payload
        payload_data = event.to_dict()
        payload_bytes = json.dumps(payload_data, separators=(",", ":")).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Furrow/1.0",
        }

        # Add signature if secret is configured
        if self.secret:
            signature = self._sign_payload(payload_bytes)
            headers["X-Furrow-Signature"] = f"sha256={signature}"

        try:
            response = await client.post(
                self.url,
                content=payload_bytes,
                headers=headers,
            )
            response.raise_for_status()
            logger.debug("Webhook sent successfully", event_type=event.type.value if isinstance(event.type, EventType) else str(event.type))
            return True
        except httpx.TimeoutException:
            logger.warning("Webhook timeout", url=self.url)
            return False
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Webhook failed with status",
                url=self.url,
                status=exc.response.status_code,
                response=exc.response.text[:200],
            )
            return False
        except Exception as exc:
            logger.warning("Webhook error", url=self.url, error=str(exc))
            return False


def create_webhook_client(settings: Settings) -> WebhookClient | None:
    """Create a webhook client from settings, or None if not configured."""
    url = getattr(settings, "webhook_url", None)
    if not url:
        return None
    secret = getattr(settings, "webhook_secret", None)
    events = getattr(settings, "webhook_events", None)
    timeout = getattr(settings, "request_timeout", 10.0)
    return WebhookClient(url=url, secret=secret, events=events, timeout=timeout)