from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import aiofiles
import httpx
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_base,
    stop_after_attempt,
    wait_exponential_jitter,
)

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."
MAX_TOKENS = 4096
HTTP_TIMEOUT_SECONDS = 120.0

# Retry policy for provider API calls. Only transient failures are retried.
RETRY_ATTEMPTS = 3
RETRY_INITIAL_WAIT = 0.1
RETRY_MAX_WAIT = 2.0
RETRY_JITTER = 1.0

# Paths that are never useful to read or list.
EXCLUDED_NAMES = frozenset({"__pycache__", "node_modules", ".git", ".venv", "venv", ".mypy_cache"})
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})

_RETRYABLE_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    httpx.TransportError,
    httpx.TimeoutException,
)
_RETRYABLE_SDK_ERRORS: tuple[type[BaseException], ...] = (
    AnthropicAPIConnectionError,
    AnthropicRateLimitError,
    OpenAIAPIConnectionError,
    OpenAIRateLimitError,
)
_SDK_STATUS_ERRORS: tuple[type[BaseException], ...] = (
    AnthropicAPIStatusError,
    OpenAIAPIStatusError,
)

_T = TypeVar("_T")


def _is_transient_failure(exc: BaseException | None) -> bool:
    """Return True only for failures that are worth retrying."""
    if exc is None:
        return False
    # Never swallow task cancellation.
    if isinstance(exc, asyncio.CancelledError):
        return False
    # httpx.TimeoutException is a subclass of httpx.TransportError.
    if isinstance(exc, _RETRYABLE_TRANSPORT_ERRORS):
        return True
    if isinstance(exc, _RETRYABLE_SDK_ERRORS):
        return True
    if isinstance(exc, _SDK_STATUS_ERRORS):
        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and status_code >= 500
    return False


class _RetryTransientFailures(retry_base):
    """Tenacity retry strategy: retry only transient provider failures."""

    def __call__(self, retry_state: RetryCallState) -> bool:
        outcome = retry_state.outcome
        if outcome is None or not outcome.failed:
            return False
        return _is_transient_failure(outcome.exception())


def _text_blocks(content: Iterable[Any]) -> list[str]:
    """Extract the text of every ``text`` block in a provider response."""
    texts: list[str] = []
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)
    return texts


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
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

    async def _with_retry(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``operation``, retrying transient provider failures."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            wait=wait_exponential_jitter(
                initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT, jitter=RETRY_JITTER
            ),
            retry=_RetryTransientFailures(),
            reraise=True,
        ):
            with attempt:
                return await operation()
        raise RuntimeError("retry loop exited without a result")

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._with_retry(lambda: self._complete_anthropic(prompt, system, model))
        if provider == Provider.OPENAI:
            return await self._with_retry(lambda: self._complete_openai(prompt, system, model))
        if provider == Provider.OLLAMA:
            return await self._with_retry(lambda: self._complete_ollama(prompt, system, model))
        raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        texts = _text_blocks(response.content)
        if not texts:
            raise ValueError("Anthropic response contained no text content block")
        return "\n\n".join(texts)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        message = response.choices[0].message
        return message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        if not isinstance(message, dict):
            raise ValueError("Ollama response contained no message object")
        content = message.get("content")
        return content if isinstance(content, str) else ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(
        self,
        directory: str | Path,
        *,
        exclude: Iterable[str] = (),
        max_results: int | None = None,
    ) -> list[str]:
        """List visible project files under ``directory`` as sorted relative paths.

        Hidden entries, common build/VCS directories and compiled artifacts are
        skipped. ``exclude`` accepts bare names (``"dist"``) or relative paths
        (``"src/generated"``); ``max_results`` truncates the sorted result.
        """
        root = Path(directory)
        if not root.is_dir():
            return []

        exclusions = {item.strip().strip("/") for item in exclude if str(item).strip()}

        def is_excluded(relative: str) -> bool:
            # Match exact relative path, prefix (directory), or bare name (basename) at any depth
            return any(
                relative == item
                or relative.startswith(f"{item}/")
                or relative.split("/")[-1] == item
                for item in exclusions
            )

        collected: list[str] = []
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            parts = relative.parts
            if any(part.startswith(".") or part in EXCLUDED_NAMES for part in parts):
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            if not path.is_file():
                continue
            relative_posix = relative.as_posix()
            if is_excluded(relative_posix):
                continue
            collected.append(relative_posix)

        results = sorted(collected)
        if max_results is not None:
            return results[: max(0, max_results)]
        return results
