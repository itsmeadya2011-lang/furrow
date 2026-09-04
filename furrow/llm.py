from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from furrow.config import Provider, Settings, settings
from furrow.logging import get_logger

logger = get_logger("llm")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (openai.RateLimitError, anthropic.RateLimitError)):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and status in (429, 500, 502, 503, 504):
        return True
    if hasattr(openai, "APIConnectionError") and isinstance(exc, openai.APIConnectionError):
        return True
    if hasattr(anthropic, "APIConnectionError") and isinstance(exc, anthropic.APIConnectionError):
        return True
    return False


retry_decorator = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._openai_ollama: AsyncOpenAI | None = None

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
    def openai_ollama(self) -> AsyncOpenAI:
        if self._openai_ollama is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY") or "ollama"
            self._openai_ollama = AsyncOpenAI(api_key=api_key, base_url=self.settings.ollama_base_url)
        return self._openai_ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None, request_timeout: float = 120.0) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        logger.debug("llm_request_started", provider=provider, model=model)
        try:
            if provider == Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model, request_timeout)
            elif provider == Provider.OPENAI:
                return await self._complete_openai(prompt, system, model, request_timeout)
            elif provider == Provider.OLLAMA:
                return await self._complete_ollama(prompt, system, model, request_timeout)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
        except Exception as e:
            logger.error("llm_request_failed", provider=provider, model=model, error=str(e))
            raise

    @retry_decorator
    async def _complete_anthropic(self, prompt: str, system: str, model: str, request_timeout: float) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
            timeout=request_timeout,
        )
        return response.content[0].text

    @retry_decorator
    async def _complete_openai(self, prompt: str, system: str, model: str, request_timeout: float) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            timeout=request_timeout,
        )
        return response.choices[0].message.content or ""

    @retry_decorator
    async def _complete_ollama(self, prompt: str, system: str, model: str, request_timeout: float) -> str:
        response = await self.openai_ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            timeout=request_timeout,
        )
        return response.choices[0].message.content or ""

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
