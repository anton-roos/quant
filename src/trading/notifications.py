"""
Lightweight notification system for the trading bot.

Supports Discord / Slack / Telegram (or any service) via webhooks.
Falls back silently when no webhook URL is configured.
"""

import logging
import threading
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class Notifier:
    """Fire-and-forget webhook notifications.

    Sends are dispatched in a daemon thread so they never block the trading loop.
    If ``webhook_url`` is empty or None, all calls are no-ops.
    """

    def __init__(self, webhook_url: Optional[str] = None, timeout: int = 10):
        self.webhook_url = (webhook_url or "").strip()
        self.timeout = timeout
        self.enabled = bool(self.webhook_url)
        if self.enabled:
            logger.info(f"Notifier enabled – webhook configured")
        else:
            logger.info("Notifier disabled – no webhook URL configured")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, message: str):
        """Send a text notification (non-blocking)."""
        if not self.enabled:
            return
        t = threading.Thread(target=self._post, args=(message,), daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _post(self, message: str):
        """POST message to the webhook URL.

        Auto-detects the payload format:
        - Discord / Slack: ``{"content": "..."}``
        - Telegram Bot API: ``{"chat_id": "<id>", "text": "..."}``
          (requires the chat_id to be embedded in the URL query string)
        """
        try:
            url = self.webhook_url

            if "api.telegram.org" in url:
                # Telegram Bot API: POST /sendMessage
                payload = {"text": message, "parse_mode": "Markdown"}
                # chat_id expected in query string already (e.g., ?chat_id=12345)
                # or we parse it from the URL
                if "chat_id=" not in url:
                    logger.warning("Telegram webhook URL missing chat_id parameter")
                    return
            elif "hooks.slack.com" in url:
                payload = {"text": message}
            else:
                # Discord / generic
                payload = {"content": message}

            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code >= 400:
                logger.debug(f"Webhook returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.debug(f"Notification failed (non-critical): {e}")
