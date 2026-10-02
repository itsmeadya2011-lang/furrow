from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as _default_settings

logger = structlog.get_logger()

# Transient errors that warrant a retry (rate limits, timeouts, 5xx).
_RETRYABLE_EXC = (
    httpx.HTTPStatusError,
    httpx.RequestError,  # connection refused, DNS failure, etc.
    TimeoutError,
)


class LLMClient:
    def __init__(self, settings: Settings | None = None, *, max_retries: int = 3) -> None:
        self.settings = settings or _default_settings
        self.max_retries = max_retries
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
        provider = self.settings.provider
        logger.debug("llm_complete", provider=provider.value, model=model)

        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(_RETRYABLE_EXC),
            reraise=True,
        )

        if provider == Provider.ANTHROPIC:
            return await retryer(self._complete_anthropic, prompt, system, model)
        elif provider == Provider.OPENAI:
            return await retryer(self._complete_openai, prompt, system, model)
        elif provider == Provider.OLLAMA:
            return await retryer(self._complete_ollama, prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        content_block = response.content[0]
        if isinstance(content_block, TextBlock):
            return content_block.text
        return ""

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
        """Call Ollama's native `/api/chat` endpoint.

        Ollama is served on ``Settings.ollama_base_url`` (default
        ``http://localhost:11434``).  Unlike Anthropic/OpenAI, Ollama returns
        the full response in a single request (unless streaming is requested).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=60) as http_client:
            resp = await http_client.post(
                f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

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
