from __future__ import annotations

import os
from pathlib import Path

import aiofiles  # type: ignore[import-untyped]  # aiofiles ships no type stubs
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as _default_settings

# Directory names that are never worth showing to an agent. Any component that
# starts with "." is skipped as well (``.git``, ``.env``, ``.venv`` ...).
_IGNORED_NAMES = frozenset(
    {"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}
)

# Substrings of provider exception class names that mark a retryable condition.
_TRANSIENT_NAME_MARKERS = (
    "ratelimit",
    "apiconnection",
    "apitimeout",
    "timeout",
    "internalserver",
    "overloaded",
    "serviceunavailable",
)

# HTTP statuses that are worth another attempt (429 throttling, 408, 5xx).
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429})


def _is_transient(exc: BaseException) -> bool:
    """Duck-typed check for errors that are worth retrying.

    Deliberately avoids importing the SDK exception classes: a provider
    exception is considered transient when it carries a retryable ``status_code``
    or when its class name matches a known transient family. Configuration
    errors (``ValueError`` raised locally) fail fast.
    """
    if isinstance(exc, ValueError):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS or status >= 500
    name = type(exc).__name__.lower()
    return any(marker in name for marker in _TRANSIENT_NAME_MARKERS)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        # Agents pass their own optional ``settings``; ``None`` must fall back to
        # the global singleton, otherwise every attribute access below raises.
        self.settings = settings if settings is not None else _default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """Client for a local Ollama server (OpenAI-compatible, no real key)."""
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        attempts = max(1, self.settings.max_retries + 1)
        retryer = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=0.5, max=8.0),
            retry=retry_if_exception(_is_transient),
            reraise=True,
        )
        return await retryer(self._complete_once, prompt, system, model)

    async def _complete_once(self, prompt: str, system: str, model: str) -> str:
        match self.settings.provider:
            case Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            case Provider.OPENAI:
                return await self._complete_openai(prompt, system, model)
            case Provider.OLLAMA:
                # There is no ollama_model setting: a local server serves one model.
                return await self._complete_openai(
                    prompt, system, self.settings.model, client=self.ollama
                )
            case _:
                raise ValueError(f"Unsupported provider: {self.settings.provider!r}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # The content is a list of block unions; a thinking or tool-use block can
        # sit at index 0, so read the first text-shaped block instead of [0].
        for block in response.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                return text
        return ""

    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        client: AsyncOpenAI | None = None,
    ) -> str:
        api = client if client is not None else self.openai
        response = await api.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """Sorted POSIX-style relative paths of the interesting files below ``directory``."""
        root = Path(directory)
        if not root.exists():
            return []
        found: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(self._is_ignored(part) for part in relative.parts):
                continue
            found.append(relative.as_posix())
        return sorted(found)

    @staticmethod
    def _is_ignored(part: str) -> bool:
        return part in _IGNORED_NAMES or part.startswith(".")
