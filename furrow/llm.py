from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings


class LLMError(Exception):
    """Base exception for LLM-related errors."""


class LLMConnectionError(LLMError):
    """Raised when a connection to the LLM provider fails."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider returns a rate limit (429) error."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""


def _is_retryable_exception(exc: BaseException) -> bool:
    """Determine if an exception should trigger a retry."""
    # Built-in transient errors
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    # Anthropic-specific errors
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500

    # OpenAI-specific errors
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500

    return False


def _classify_exception(exc: BaseException) -> LLMError:
    """Classify an exception into a structured LLM error."""
    if isinstance(exc, (TimeoutError, anthropic.APITimeoutError, openai.APITimeoutError)):
        return LLMTimeoutError(f"LLM request timed out: {exc}") from exc
    if isinstance(exc, anthropic.RateLimitError) or isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError(f"LLM rate limit exceeded: {exc}") from exc
    if isinstance(exc, (ConnectionError, anthropic.APIConnectionError, openai.APIConnectionError)):
        return LLMConnectionError(f"LLM connection error: {exc}") from exc
    return LLMError(f"LLM request failed: {exc}") from exc


class LLMClient:
    def __init__(
        self,
        settings: Settings = settings,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
    ) -> None:
        self.settings = settings
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    def _get_retryer(self) -> AsyncRetrying:
        """Create a tenacity AsyncRetrying instance with the configured settings."""
        return AsyncRetrying(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=self.retry_min_wait, max=self.retry_max_wait),
            retry=retry_if_exception(_is_retryable_exception),
            reraise=False,
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

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        last_exc: BaseException | None = None
        async for attempt in self._get_retryer():
            with attempt:
                try:
                    response = await self.anthropic.messages.create(
                        model=model,
                        max_tokens=4096,
                        system=system or "You are a helpful coding assistant.",
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text
                except (anthropic.APIError, ConnectionError, TimeoutError) as exc:
                    last_exc = exc
                    raise
        raise _classify_exception(last_exc) if last_exc else LLMError("Unknown error")

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        last_exc: BaseException | None = None
        async for attempt in self._get_retryer():
            with attempt:
                try:
                    response = await self.openai.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system or "You are a helpful coding assistant."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return response.choices[0].message.content or ""
                except (openai.APIError, ConnectionError, TimeoutError) as exc:
                    last_exc = exc
                    raise
        raise _classify_exception(last_exc) if last_exc else LLMError("Unknown error")

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
