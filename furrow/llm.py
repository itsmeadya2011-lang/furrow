"""Asynchronous LLM client with provider dispatch, retries, and file helpers.

The module imports the Anthropic and OpenAI SDKs eagerly (both are hard declared
dependencies) but imports `httpx` lazily, because only the Ollama provider needs
it.  `tenacity` is imported defensively: if it is missing a small stdlib-only
retry loop takes over with identical semantics.
"""

from __future__ import annotations

import asyncio
import fnmatch
import importlib
import os
from collections.abc import Awaitable, Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings as default_settings

__all__ = ["LLMClient", "LLMError"]

_MAX_RETRY_WAIT_SECONDS = 30.0
_TRUNCATION_MARKER = "\n... [truncated]"


class LLMError(RuntimeError):
    """Raised when an LLM provider call fails or returns unusable output."""


def _transient_errors() -> tuple[type[BaseException], ...]:
    """Collect retryable provider exception classes available in this environment.

    Returns:
        The exception classes that should be retried.  The tuple is empty when
        neither SDK is importable or exposes none of the known classes.
    """
    candidates: list[type[BaseException]] = []
    retryable_names = (
        "RateLimitError",
        "APIConnectionError",
        "InternalServerError",
        "APITimeoutError",
    )
    for module_name in ("anthropic", "openai"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for class_name in retryable_names:
            cls = getattr(module, class_name, None)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                candidates.append(cls)
    return tuple(candidates)


TRANSIENT_ERRORS: tuple[type[BaseException], ...] = _transient_errors()

_tenacity: Any
try:  # pragma: no cover - depends on the installed environment
    import tenacity

    _tenacity = tenacity
except ImportError:  # pragma: no cover - exercised only without tenacity
    _tenacity = None


class LLMClient:
    """Async wrapper around the supported providers plus workspace file helpers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings if settings is not None else default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: Any | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Return the lazily created Anthropic client.

        Raises:
            ValueError: If no Anthropic API key is configured.
        """
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Return the lazily created OpenAI client.

        Raises:
            ValueError: If no OpenAI API key is configured.
        """
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def _http(self) -> Any:
        """Return a lazily created ``httpx.AsyncClient`` for native provider calls.

        Raises:
            LLMError: If `httpx` is not installed.
        """
        if self._httpx is None:
            try:
                httpx = importlib.import_module("httpx")
            except ImportError as exc:
                raise LLMError(
                    "httpx is required for the ollama provider but is not installed"
                ) from exc
            self._httpx = httpx.AsyncClient(timeout=self.settings.request_timeout)
        return self._httpx

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Send `prompt` to the configured provider and return the text reply.

        Args:
            prompt: The user prompt.
            system: Optional system prompt; falls back to the configured default.
            model: Optional model override; falls back to ``settings.model``.

        Returns:
            The assistant response text.

        Raises:
            LLMError: If the provider returns unusable output.
            ValueError: If the configured provider is not supported.
        """
        resolved = model or self.settings.model
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            operation = partial(self._complete_anthropic, prompt, system, resolved)
        elif provider == Provider.OPENAI:
            operation = partial(self._complete_openai, prompt, system, resolved)
        elif provider == Provider.OLLAMA:
            operation = partial(self._complete_ollama, prompt, system, resolved)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        return await self._with_retry(operation)

    async def _with_retry(self, operation: Callable[[], Awaitable[str]]) -> str:
        """Run `operation`, retrying transient provider errors with backoff.

        The attempt count and base delay are read from settings at call time, so
        the retry policy always reflects the live configuration.

        Args:
            operation: A zero-argument callable returning the awaitable response.

        Returns:
            The result of the first successful attempt.

        Raises:
            BaseException: The last provider error when every attempt fails.
        """
        attempts = max(1, self.settings.max_retries)
        if _tenacity is None:
            return await self._manual_retry(operation, attempts)
        async for attempt in _tenacity.AsyncRetrying(
            stop=_tenacity.stop_after_attempt(attempts),
            wait=_tenacity.wait_exponential(
                multiplier=self.settings.retry_base_delay,
                max=_MAX_RETRY_WAIT_SECONDS,
            ),
            retry=_tenacity.retry_if_exception_type(TRANSIENT_ERRORS),
            reraise=True,
        ):
            with attempt:
                return await operation()
        raise LLMError("retry loop finished without producing a result")

    async def _manual_retry(
        self,
        operation: Callable[[], Awaitable[str]],
        attempts: int,
    ) -> str:
        """Retry loop used when tenacity is unavailable.

        Args:
            operation: A zero-argument callable returning the awaitable response.
            attempts: Maximum number of attempts, including the first one.

        Returns:
            The result of the first successful attempt.

        Raises:
            BaseException: The last provider error when every attempt fails.
        """
        for attempt in range(1, attempts + 1):
            try:
                return await operation()
            except TRANSIENT_ERRORS:
                if attempt >= attempts:
                    raise
                await asyncio.sleep(self._backoff_seconds(attempt))
        raise LLMError("retry loop finished without producing a result")

    def _backoff_seconds(self, attempt: int) -> float:
        """Return the capped exponential backoff delay before retrying `attempt`."""
        delay = self.settings.retry_base_delay * (2 ** (attempt - 1))
        return min(delay, _MAX_RETRY_WAIT_SECONDS)

    @staticmethod
    def _anthropic_text(response: Any) -> str:
        """Join every text block of an Anthropic message response.

        Args:
            response: The SDK response object for a message creation call.

        Returns:
            The concatenated text of all ``text`` content blocks.

        Raises:
            LLMError: If the response carries no non-empty text block.
        """
        blocks = getattr(response, "content", None) or []
        parts: list[str] = []
        for block in blocks:
            if getattr(block, "type", None) != "text":
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        joined = "".join(parts)
        if not joined.strip():
            stop_reason = getattr(response, "stop_reason", "unknown")
            raise LLMError(
                "Anthropic returned no text content "
                f"(stop_reason={stop_reason!r}, blocks={len(blocks)})"
            )
        return joined

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        """Complete `prompt` through the Anthropic messages API."""
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or self.settings.default_system_prompt,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.settings.request_timeout,
        )
        return self._anthropic_text(response)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        """Complete `prompt` through the OpenAI chat completions API.

        Raises:
            LLMError: If the response carries no choices at all.
        """
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system or self.settings.default_system_prompt},
                {"role": "user", "content": prompt},
            ],
            timeout=self.settings.request_timeout,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("OpenAI returned a response without any choices")
        content = getattr(choices[0].message, "content", None)
        return "" if content is None else str(content)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Complete `prompt` through the native Ollama chat API.

        Raises:
            LLMError: If the request fails or the payload carries no content.
        """
        client = self._http
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or self.settings.default_system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama request to {url} failed: {exc}") from exc
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if content is None:
            raise LLMError(f"Ollama response from {url} has no message.content field")
        text = content if isinstance(content, str) else str(content)
        if not text.strip():
            raise LLMError(f"Ollama response from {url} has empty message.content")
        return text

    async def aclose(self) -> None:
        """Release lazily created network resources; safe to call repeatedly."""
        client, self._httpx = self._httpx, None
        if client is not None:
            await client.aclose()

    async def read_file(self, path: str | Path) -> str:
        """Read a UTF-8 text file.

        Raises:
            FileNotFoundError: If `path` does not exist.
        """
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        async with aiofiles.open(target, "r", encoding="utf-8") as handle:
            return str(await handle.read())

    async def read_files(
        self,
        paths: Iterable[str | Path],
        *,
        max_chars: int = 4000,
    ) -> dict[str, str]:
        """Read several files concurrently, truncating each to `max_chars`.

        Unreadable paths are omitted from the result instead of raising.

        Args:
            paths: The files to load.
            max_chars: Maximum characters kept per file; ``0`` disables truncation.

        Returns:
            A mapping of path string to (possibly truncated) file content.
        """
        wanted = [Path(item) for item in paths]
        if not wanted:
            return {}
        results = await asyncio.gather(
            *(self.read_file(item) for item in wanted),
            return_exceptions=True,
        )
        contents: dict[str, str] = {}
        for item, result in zip(wanted, results):
            if isinstance(result, BaseException):
                continue
            text = str(result)
            if max_chars > 0 and len(text) > max_chars:
                text = text[:max_chars] + _TRUNCATION_MARKER
            contents[str(item)] = text
        return contents

    async def write_file(self, path: str | Path, content: str) -> Path:
        """Write `content` as UTF-8 text, creating parent directories as needed.

        Returns:
            The path that was written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as handle:
            await handle.write(content)
        return target

    @staticmethod
    def _is_ignored(relative: Path, ignore_patterns: tuple[str, ...]) -> bool:
        """Return `True` when any parent directory of `relative` must be pruned."""
        return any(
            fnmatch.fnmatch(part, pattern)
            for part in relative.parts[:-1]
            for pattern in ignore_patterns
        )

    def list_files(
        self,
        directory: str | Path,
        *,
        pattern: str | None = None,
    ) -> list[str]:
        """List workspace files as deterministic, sorted, relative paths.

        Directories matching ``settings.ignore_dirs`` (exact name or glob) are
        pruned, and files larger than ``settings.max_file_bytes`` are skipped.
        Callers that build prompts should cap the result at
        ``settings.max_prompt_files`` entries and read the contents with
        ``read_files(..., max_chars=settings.file_context_chars)``.

        Args:
            directory: The root to walk.
            pattern: Optional `fnmatch` pattern applied to the relative path.

        Returns:
            Sorted relative paths; empty when `directory` does not exist.
        """
        root = Path(directory)
        if not root.is_dir():
            return []
        ignore_patterns = tuple(self.settings.ignore_dirs)
        found: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if self._is_ignored(relative, ignore_patterns):
                continue
            if pattern is not None and not fnmatch.fnmatch(relative.as_posix(), pattern):
                continue
            try:
                if path.stat().st_size > self.settings.max_file_bytes:
                    continue
            except OSError:
                continue
            found.append(str(relative))
        return sorted(found)