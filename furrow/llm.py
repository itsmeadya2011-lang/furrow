from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings


def _build_retryable_exceptions() -> tuple[type[BaseException], ...]:
    """Build the tuple of retryable exception types defensively.

    Each SDK is imported lazily so the module still imports cleanly even if a
    given SDK is absent from the environment.
    """
    retryable: list[type[BaseException]] = [asyncio.TimeoutError]

    try:
        import anthropic as _anthropic
    except ImportError:  # pragma: no cover - anthropic is a hard dependency
        _anthropic = None  # type: ignore[assignment]

    if _anthropic is not None:
        retryable.extend(
            exc
            for exc in (
                getattr(_anthropic, "APIConnectionError", None),
                getattr(_anthropic, "RateLimitError", None),
            )
            if exc is not None
        )

    try:
        import openai as _openai
    except ImportError:  # pragma: no cover - openai is a hard dependency
        _openai = None  # type: ignore[assignment]

    if _openai is not None:
        retryable.extend(
            exc
            for exc in (
                getattr(_openai, "APIConnectionError", None),
                getattr(_openai, "RateLimitError", None),
            )
            if exc is not None
        )

    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover - httpx is a hard dependency
        _httpx = None  # type: ignore[assignment]

    if _httpx is not None:
        retryable.extend(
            exc
            for exc in (
                getattr(_httpx, "TransportError", None),
                getattr(_httpx, "TimeoutException", None),
            )
            if exc is not None
        )

    return tuple(retryable)


_RETRYABLE_EXCEPTIONS = _build_retryable_exceptions()

_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

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

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @_RETRY_DECORATOR
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return ""
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

    @_RETRY_DECORATOR
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @_RETRY_DECORATOR
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        base_url = self.settings.ollama_base_url
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
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