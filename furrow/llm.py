from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

try:  # pragma: no cover - exercised implicitly by import-time behavior
    import anthropic
    from anthropic import AsyncAnthropic
except ImportError:  # pragma: no cover - SDK not installed / version drift
    anthropic = None  # type: ignore[assignment]
    AsyncAnthropic = None  # type: ignore[assignment,misc]

try:  # pragma: no cover - exercised implicitly by import-time behavior
    import openai
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - SDK not installed / version drift
    openai = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment,misc]


def _build_transient_errors() -> tuple[type[BaseException], ...]:
    """Collect transport + SDK rate-limit/connection/timeout exception classes.

    Auth, validation and bad-request errors are deliberately excluded -- retrying
    them only burns time. Classification uses ``isinstance``, so subclasses of
    the collected classes (e.g. ``httpx.ConnectError``) are transient too.
    """
    errors: list[type[BaseException]] = [httpx.TransportError]
    for sdk in (anthropic, openai):
        if sdk is None:
            continue
        for name in ("RateLimitError", "APIConnectionError", "APITimeoutError"):
            candidate = getattr(sdk, name, None)
            if (
                isinstance(candidate, type)
                and issubclass(candidate, BaseException)
                and candidate not in errors
            ):
                errors.append(candidate)
    return tuple(errors)


TRANSIENT_ERRORS: tuple[type[BaseException], ...] = _build_transient_errors()

RETRY_KWARGS: dict[str, Any] = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=1, max=10),
}


def is_transient_error(exc: BaseException) -> bool:
    """Return True when ``exc`` is worth retrying (see TRANSIENT_ERRORS)."""
    return isinstance(exc, TRANSIENT_ERRORS)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            if AsyncAnthropic is None:
                raise RuntimeError(
                    "The 'anthropic' package is not installed; cannot use the anthropic provider"
                )
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            if AsyncOpenAI is None:
                raise RuntimeError(
                    "The 'openai' package is not installed; cannot use the openai provider"
                )
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        retrier = AsyncRetrying(
            retry=retry_if_exception(is_transient_error),
            reraise=True,
            **RETRY_KWARGS,
        )
        return await retrier(self._complete_once, prompt, system, model)

    async def _complete_once(self, prompt: str, system: str, model: str) -> str:
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Only text blocks carry output; thinking/tool blocks have no `.text`.
        texts = [getattr(block, "text", None) for block in response.content]
        joined = "".join(text for text in texts if isinstance(text, str))
        if not joined:
            raise ValueError("Anthropic response did not contain any text content")
        return joined

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        timeout = httpx.Timeout(120.0, connect=10.0)
        client_kwargs: dict[str, Any] = {
            "base_url": self.settings.ollama_base_url,
            "timeout": timeout,
        }
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system or "You are a helpful coding assistant.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
            )
        response.raise_for_status()
        data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError(
                f"Ollama response did not contain message.content: {data!r}. "
                f"Is '{self.settings.ollama_base_url}' a valid Ollama server?"
            )
        return message["content"]

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]


__all__ = [
    "LLMClient",
    "RETRY_KWARGS",
    "TRANSIENT_ERRORS",
    "is_transient_error",
]
