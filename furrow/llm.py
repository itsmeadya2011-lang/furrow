from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from furrow.config import Provider, Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

# Transient provider failures only: auth/bad-request errors will never succeed on a retry, and a
# missing API key (ValueError) must not be retried either.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class LLMError(ValueError):
    """Raised when an LLM request cannot be completed.

    Subclasses ValueError so existing `except ValueError` handlers keep working.
    """


def _is_retryable_status(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        return False
    return status in RETRYABLE_STATUS_CODES or status >= 500


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            # max_retries=0: tenacity below owns the retry budget so it is not silently tripled.
            self._anthropic = AsyncAnthropic(api_key=api_key, max_retries=0)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key, max_retries=0)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        # Ollama serves an OpenAI-compatible API, so the OpenAI client talks to it unchanged and
        # only the base URL (and a placeholder key, which Ollama ignores) differs.
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                api_key="ollama",
                base_url=self.settings.ollama_base_url,
                max_retries=0,
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        try:
            provider = Provider(self.settings.provider)
        except ValueError as exc:
            raise LLMError(f"Unsupported provider: {self.settings.provider!r}") from exc

        if provider is Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._complete_openai(prompt, system, model)
        raise LLMError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        return await self._with_retry(self._create_anthropic, prompt, system, model)

    async def _create_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.settings.request_timeout,
        )
        # Extended thinking and tool use interleave non-text blocks, so join every text block
        # instead of assuming content[0] is text.
        texts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if text:
                    texts.append(text)
        if not texts:
            raise LLMError(f"Anthropic response contained no text block (model={model})")
        return "".join(texts)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        return await self._with_retry(self._create_openai, prompt, system, model)

    async def _create_openai(self, prompt: str, system: str, model: str) -> str:
        client = self._openai_client()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            timeout=self.settings.request_timeout,
        )
        if not response.choices:
            raise LLMError(f"OpenAI response contained no choices (model={model})")
        content = response.choices[0].message.content
        if not content:
            raise LLMError(f"OpenAI response contained no message content (model={model})")
        return content

    def _openai_client(self) -> AsyncOpenAI:
        if Provider(self.settings.provider) is Provider.OLLAMA:
            return self.ollama
        return self.openai

    def _attempts(self) -> int:
        return max(1, int(self.settings.max_retries or 0))

    def _before_sleep(self, state: RetryCallState) -> None:
        outcome = state.outcome
        error = outcome.exception() if outcome is not None else None
        sleep = getattr(state.next_action, "sleep", 0.0)
        logger.warning(
            "LLM request failed on attempt %s/%s (%r); retrying in %.1fs",
            state.attempt_number,
            self._attempts(),
            error,
            sleep,
        )

    async def _with_retry(self, call: Callable[..., Awaitable[str]], *args: object) -> str:
        attempts = self._attempts()
        if attempts == 1:
            return await call(*args)
        retrying = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0) + wait_random(0.0, 1.0),
            retry=retry_if_exception_type(TRANSIENT_ERRORS)
            | retry_if_exception(_is_retryable_status),
            before_sleep=self._before_sleep,
            reraise=False,
        )
        try:
            return await retrying(call, *args)
        except RetryError as exc:
            last = exc.last_attempt.exception() if exc.last_attempt is not None else exc
            raise LLMError(f"LLM request failed after {attempts} attempt(s): {last}") from last

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        async with aiofiles.open(p) as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        root = Path(directory)
        if not root.exists():
            return []
        files: list[str] = []
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRECTORIES)
            base = Path(current)
            for name in filenames:
                if name.endswith(".pyc"):
                    continue
                files.append((base / name).relative_to(root).as_posix())
        return sorted(files)
