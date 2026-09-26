from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
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


class LLMError(Exception):
    """Raised when an LLM request fails after exhausting retries."""


class LLMClient:
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
            self._httpx = httpx.AsyncClient(timeout=60.0)
        return self._httpx

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider

        if provider == Provider.ANTHROPIC:
            fn = self._complete_anthropic
        elif provider == Provider.OPENAI:
            fn = self._complete_openai
        elif provider == Provider.OLLAMA:
            fn = self._complete_ollama
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return await self._with_retry(fn, prompt, system, model)

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        """Run a completion coroutine with tenacity-powered retry on transient errors."""
        attempts = self.settings.retry_max_attempts
        if attempts <= 1:
            return await fn(*args, **kwargs)

        logger.debug("llm_retry_config", attempts=attempts)
        retrying = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=self.settings.retry_backoff_min,
                min=self.settings.retry_backoff_min,
                max=self.settings.retry_backoff_max,
            ),
            retry=retry_if_exception_type((httpx.HTTPError, LLMError)),
            reraise=True,
        )
        return await retrying(fn, *args, **kwargs)

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            logger.debug("llm_anthropic_complete", model=model, characters=len(text))
            return text
        except Exception as e:
            raise LLMError(f"Anthropic request failed: {e}") from e

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content or ""
            logger.debug("llm_openai_complete", model=model, characters=len(text))
            return text
        except Exception as e:
            raise LLMError(f"OpenAI request failed: {e}") from e

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            }
            resp = await self.httpx_client.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            logger.debug("llm_ollama_complete", model=model, characters=len(text))
            return text
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e

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
