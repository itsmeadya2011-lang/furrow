from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)

# Collect retryable exception types from each provider library.
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    TimeoutError,
)
try:
    _RETRYABLE_EXCEPTIONS += (anthropic.APIError,)
except AttributeError:  # pragma: no cover
    pass
try:
    _RETRYABLE_EXCEPTIONS += (openai.APIError,)
except AttributeError:  # pragma: no cover
    pass


class LLMClient:
    """Async LLM client supporting Anthropic, OpenAI, and Ollama providers."""

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
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
            self._httpx = httpx.AsyncClient(base_url=self.settings.ollama_base_url)
        return self._httpx

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        logger.debug("llm_request", provider=provider.value, model=model)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                return await self._dispatch(prompt, system, model, provider)

    async def _dispatch(
        self,
        prompt: str,
        system: str,
        model: str,
        provider: Provider,
    ) -> str:
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content and len(response.content) > 0:
            return response.content[0].text
        raise ValueError("Empty response from Anthropic")

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content
        raise ValueError("Empty response from OpenAI")

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
        }
        response = await self.httpx_client.post("/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

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
