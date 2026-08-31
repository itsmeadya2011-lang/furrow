from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if the exception is a transient error worth retrying.

    Retries on API errors and timeouts, but not on authentication failures.
    """
    auth_errors = (
        anthropic.AuthenticationError,
        openai.AuthenticationError,
    )
    if isinstance(exc, auth_errors):
        return False
    transient_errors = (
        anthropic.APIError,
        anthropic.APITimeoutError,
        openai.APIError,
        openai.APITimeoutError,
        TimeoutError,
        ConnectionError,
    )
    return isinstance(exc, transient_errors)


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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(_is_transient_error),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        start_time = time.monotonic()
        try:
            if self.settings.provider == Provider.ANTHROPIC:
                result = await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                result = await self._complete_openai(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")
            latency = time.monotonic() - start_time
            logger.info(
                "llm_call_completed",
                model=model,
                provider=self.settings.provider.value,
                latency_ms=round(latency * 1000, 2),
                prompt_tokens=len(prompt),
                response_tokens=len(result),
            )
            return result
        except Exception as e:
            latency = time.monotonic() - start_time
            logger.error(
                "llm_call_failed",
                model=model,
                provider=self.settings.provider.value,
                latency_ms=round(latency * 1000, 2),
                error=str(e),
            )
            raise

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
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
