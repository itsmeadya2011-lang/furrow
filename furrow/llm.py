from __future__ import annotations

import logging
import os
from pathlib import Path

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = logging.getLogger(__name__)

# Errors that are worth retrying on (transient / rate-limit issues).
_RETRYABLE_OPENAI = (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError)

# The anthropic SDK's exception hierarchy changed across versions (the Stainless
# rewrite landed around 0.41; the project allows >=0.34, which is the new SDK).
# Build the retryable tuple defensively with getattr so module import never breaks
# on a renamed class. We retry only *transient* errors (rate limits, timeouts,
# 5xx) rather than the base, so non-retryable failures (401/400/...) surface
# immediately. If none of the specific names are present (unexpected), fall back
# to the base class so we never silently skip retrying.
_ANTHROPIC_RETRYABLE_NAMES = (
    "RateLimitError",  # 429
    "APITimeoutError",  # request timeout
    "APIConnectionError",  # connection/network errors
    "InternalServerError",  # 5xx
    "OverloadedError",  # 529
    "ServiceUnavailableError",  # 503
)
_RETRYABLE_ANTHROPIC = tuple(
    getattr(anthropic, name) for name in _ANTHROPIC_RETRYABLE_NAMES if hasattr(anthropic, name)
)
if not _RETRYABLE_ANTHROPIC and hasattr(anthropic, "AnthropicError"):
    _RETRYABLE_ANTHROPIC = (anthropic.AnthropicError,)


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
            # Disable the SDK's built-in retries so that tenacity (in `complete`)
            # is the single authority over retry/backoff behaviour and we avoid
            # compounding retries that can cause rate-limit deadlocks.
            self._anthropic = AsyncAnthropic(api_key=api_key, max_retries=0)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            # Let tenacity (in `complete`) own retry/backoff behaviour.
            self._openai = AsyncOpenAI(api_key=api_key, max_retries=0)
        return self._openai

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, max_tokens)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    # -- Per-provider completions with retry/backoff ---------------------

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        async for attempt in AsyncRetrying(
            reraise=True,
            retry=retry_if_exception_type(_RETRYABLE_ANTHROPIC),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(3),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        ):
            with attempt:
                response = await self.anthropic.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system or "You are a helpful coding assistant.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
        # Should be unreachable due to reraise
        raise RuntimeError("Failed to get completion from Anthropic")

    async def _complete_openai(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        async for attempt in AsyncRetrying(
            reraise=True,
            retry=retry_if_exception_type(_RETRYABLE_OPENAI),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            stop=stop_after_attempt(3),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        ):
            with attempt:
                response = await self.openai.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system or "You are a helpful coding assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""
        raise RuntimeError("Failed to get completion from OpenAI")

    async def _complete_ollama(
        self, prompt: str, system: str, model: str, max_tokens: int
    ) -> str:
        client = AsyncOpenAI(
            base_url=self.settings.ollama_base_url + "/v1",
            api_key="ollama",
            max_retries=0,
        )
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    # -- File utilities --------------------------------------------------

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
