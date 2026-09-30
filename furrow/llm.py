from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore[import-untyped]
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings
from furrow.logging import logger

# Retry on transient errors: timeouts, connection errors, rate limits
_RETRY_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.TransportError,
    ConnectionError,
    TimeoutError,
)


class LLMClient:
    """Client for LLM completion across Anthropic, OpenAI, and Ollama providers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

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
    def httpx_client(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=httpx.Timeout(120.0),
            )
        return self._httpx

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Send a prompt and return the LLM response, with retry on transient failures."""
        model = model or self.settings.model
        log = logger.bind(model=model, provider=self.settings.provider.value)
        log.debug("completing_llm_request")

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                response = await self._do_complete(prompt, system, model)
                log.debug("llm_request_succeeded")
                return response

        # Unreachable: retry loop always returns or raises
        raise RuntimeError("LLM completion failed after retries")

    async def _do_complete(self, prompt: str, system: str, model: str) -> str:
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
        return response.content[0].text  # type: ignore[union-attr]

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        return content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Call the Ollama API via HTTP. Ollama uses an OpenAI-compatible endpoint."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self.httpx_client.post(
            "/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return str(data["message"]["content"])

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
            content = await f.read()
            return str(content)

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

    @property
    def workspace(self) -> Path:
        return self.settings.workspace

    async def close(self) -> None:
        """Close any open HTTP client connections."""
        if self._httpx is not None:
            await self._httpx.aclose()
            self._httpx = None
