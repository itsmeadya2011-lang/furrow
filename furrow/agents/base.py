from __future__ import annotations

from pathlib import Path

from furrow.config import Settings
from furrow.config import settings as default_settings
from furrow.llm import LLMClient


class BaseAgent:
    """Client/settings resolution shared by the planner, worker and tester."""

    model_key = "model"

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        resolved = settings or default_settings
        self.client = client if client is not None else LLMClient(settings=resolved)
        self.settings: Settings = settings or getattr(client, "settings", None) or resolved

    @property
    def model(self) -> str:
        """Model for this agent's role, falling back to the default model."""
        return getattr(self.settings, self.model_key, None) or self.settings.model

    @property
    def workspace(self) -> Path:
        """Resolved per call: the configured path may be relative or stale."""
        ws = Path(self.settings.workspace).expanduser()
        if not ws.is_absolute():
            ws = Path.cwd() / ws
        return ws.resolve()
