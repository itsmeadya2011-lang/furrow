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
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    retry_if_not_result,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)

# Exceptions that are safe to retry (transient / network errors)
_RETRYABLE_EXCEPTIONS = (
    anthropic.APIError,
    anthropic.APITimeoutError,
    openai.APIError,
    openai.OpenAIError,
    httpx.TimeoutException,
    httpx.HTTPStatusError,
    asyncio.TimeoutError,
    ConnectionError,
)


def _log_retry(retry_state: Any) -> None:
    """Log retry attempts for observability."""
    if retry_state.outcome is not None:
        if retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            logger.warning(
                "llm_retry",
                attempt_number=retry_state.attempt_number,
                error=type(exc).__name__ if exc else "unknown",
                error_message=str(exc)[:200] if exc else "",
            )
        else:
            result = retry_state.outcome.result()
            logger.warning(
                "llm_retry_empty",
                attempt_number=retry_state.attempt_number,
                response_preview=(result[:200] if isinstance(result, str) else repr(result)),
            )


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
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
            )
        return self._openai

    @property
    def httpx_client(self) -> httpx.AsyncClient:
        """HTTP client for Ollama provider."""
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(
                timeout=self.settings.llm_timeout,
            )
        return self._httpx

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        """Complete a prompt with retry logic and response validation.

        Uses tenacity ``AsyncRetrying`` with configurable max retries
        (``settings.llm_max_retries``) and exponential backoff.
        Retries on retryable exceptions or empty/whitespace-only responses.
        """
        model = model or self.settings.model
        max_tokens = self.settings.default_max_tokens

        retryer = AsyncRetrying(
            stop=stop_after_attempt(self.settings.llm_max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=(
                retry_if_exception_type(_RETRYABLE_EXCEPTIONS)
                | retry_if_not_result(lambda r: bool(r) and bool(r.strip()))
            ),
            before_sleep=_log_retry,
            reraise=True,
        )
        return await retryer(
            self._complete_with_provider, prompt, system, model, max_tokens
        )

    async def _complete_with_provider(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        """Dispatch to the appropriate provider implementation."""
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, max_tokens)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        """Complete using Ollama's chat API via HTTP."""
        response = await self.httpx_client.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
        )
        response.raise_for_status()
        data = response.json()
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

    async def close(self) -> None:
        """Close all lazily-initialized HTTP clients."""
        if self._httpx is not None:
            await self._httpx.aclose()
            self._httpx = None
