from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.config import settings as _default_settings

log = structlog.get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or _default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: httpx.AsyncClient | None = None

    @staticmethod
    def retryable_errors() -> tuple[type[BaseException], ...]:
        """Exceptions that are worth retrying (transient / rate-limited)."""
        return (
            anthropic.APIError,
            openai.APIError,
            httpx.HTTPError,
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
        )

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
    def ollama(self) -> httpx.AsyncClient:
        if self._ollama is None:
            self._ollama = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=self.settings.request_timeout,
            )
        return self._ollama

    def _retrying(self) -> tenacity.AsyncRetrying:
        return tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(self.settings.max_retries or 1),
            wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
            retry=tenacity.retry_if_exception_type(self.retryable_errors()),
            reraise=True,
        )

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        max_tokens = self.settings.max_tokens
        log.debug("completing", provider=self.settings.provider, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, max_tokens)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str, max_tokens: int) -> str:
        async def call() -> str:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            # A message may contain non-text blocks (e.g. tool_use); extract the
            # first textual content so we never index into a missing attribute.
            return self._first_text(response.content)

        return await self._retrying()(call)

    @staticmethod
    def _first_text(content: Any) -> str:
        for block in content:
            text = getattr(block, "text", None)
            if text:
                return text
        return ""

    async def _complete_openai(self, prompt: str, system: str, model: str, max_tokens: int) -> str:
        async def call() -> str:
            response = await self.openai.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""

        return await self._retrying()(call)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        async def call() -> str:
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = await self.ollama.post(
                "/api/chat",
                json={"model": model, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")

        return await self._retrying()(call)

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
