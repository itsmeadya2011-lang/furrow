from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, get_settings

__all__ = [
    "LLMClient",
    "LLMConfigError",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponseError",
]


class LLMError(RuntimeError):
    """Base error raised by :class:`LLMClient`."""


class LLMConfigError(LLMError, ValueError):
    """Configuration error (missing API key, unimplemented provider, ...)."""


class LLMResponseError(LLMError):
    """A malformed, empty, or truncated provider response."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited us and every retry was exhausted."""


_DEFAULT_SYSTEM = "You are a helpful coding assistant."
_BACKOFF_MAX = 10.0


def _sdk_retryable_errors() -> tuple[type[BaseException], ...]:
    """Transient exception types from the installed providers.

    SDK-specific types are imported lazily inside this function so a future
    SDK version bump that renames or removes a type degrades to "no retries"
    instead of crashing at module import time. httpx transport errors are
    included for robustness since both SDKs surface HTTP failures through it.
    """
    errors: list[type[BaseException]] = []
    try:
        import anthropic as _a

        errors.extend(
            [
                _a.APIConnectionError,
                _a.APITimeoutError,
                _a.RateLimitError,
                _a.InternalServerError,
                _a.ServiceUnavailableError,
                _a.APIStatusError,
            ]
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        import openai as _o

        errors.extend(
            [
                _o.APIConnectionError,
                _o.APITimeoutError,
                _o.RateLimitError,
                _o.InternalServerError,
                _o.APIStatusError,
            ]
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        import httpx

        errors.extend(
            [
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.TransportError,
                httpx.HTTPStatusError,
            ]
        )
    except Exception:  # noqa: BLE001
        pass
    return tuple(errors)


def _sdk_rate_limit_types() -> tuple[type[BaseException], ...]:
    """Rate-limit exception types from the installed providers (lazy import)."""
    types: list[type[BaseException]] = []
    try:
        import anthropic as _a

        types.append(_a.RateLimitError)
    except Exception:  # noqa: BLE001
        pass
    try:
        import openai as _o

        types.append(_o.RateLimitError)
    except Exception:  # noqa: BLE001
        pass
    return tuple(types)


def _is_rate_limit(exc: BaseException) -> bool:
    for cls in _sdk_rate_limit_types():
        if isinstance(exc, cls):
            return True
    return getattr(exc, "status_code", None) == 429


class LLMClient:
    """Async client for the configured LLM provider."""

    _LIST_SKIP = frozenset(
        {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "target",
            "dist",
            "build",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            ".eggs",
        }
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMConfigError("ANTHROPIC_API_KEY is not set")
            # SDK-level retries disabled (max_retries=0): retry policy lives in
            # exactly one place -- the tenacity wrapper below.
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
                max_retries=0,
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMConfigError("OPENAI_API_KEY is not set")
            # SDK-level retries disabled (max_retries=0); see the anthropic
            # property above.
            self._openai = AsyncOpenAI(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
                max_retries=0,
            )
        return self._openai

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
    ) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        try:
            if provider == Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            if provider == Provider.OPENAI:
                return await self._complete_openai(prompt, system, model)
            raise LLMConfigError(f"Provider '{provider.value}' is not yet implemented")
        except LLMError:
            raise
        except Exception as exc:
            raise LLMResponseError(
                f"[{provider.value}/{model}] unexpected provider error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    async def _complete_anthropic(
        self, prompt: str, system: str | None, model: str
    ) -> str:
        client = self.anthropic  # may raise LLMConfigError before retrying

        async def _create() -> object:
            return await client.messages.create(
                model=model,
                max_tokens=self.settings.llm_max_tokens,
                system=system or _DEFAULT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

        response = await self._call_with_retry(
            provider=Provider.ANTHROPIC,
            model=model,
            fn=_create,
        )
        return self._extract_anthropic_text(response)

    async def _complete_openai(
        self, prompt: str, system: str | None, model: str
    ) -> str:
        client = self.openai  # may raise LLMConfigError before retrying

        async def _create() -> object:
            return await client.chat.completions.create(
                model=model,
                max_tokens=self.settings.llm_max_tokens,
                messages=[
                    {"role": "system", "content": system or _DEFAULT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )

        response = await self._call_with_retry(
            provider=Provider.OPENAI,
            model=model,
            fn=_create,
        )
        return self._extract_openai_text(response)

    async def _call_with_retry(
        self,
        provider: Provider,
        model: str,
        fn: Callable[[], Awaitable[object]],
    ) -> object:
        attempts = max(1, int(self.settings.llm_max_retries))
        retryable = _sdk_retryable_errors()
        if retryable:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=1, min=1, max=_BACKOFF_MAX),
                retry=retry_if_exception_type(retryable),
                reraise=True,
            )
        else:
            retrying = AsyncRetrying(stop=stop_after_attempt(attempts), reraise=True)
        try:
            return await retrying(fn)
        except LLMError:
            raise
        except Exception as exc:
            if _is_rate_limit(exc):
                raise LLMRateLimitError(
                    f"[{provider.value}/{model}] rate limited after "
                    f"{attempts} attempt(s)"
                ) from exc
            raise LLMResponseError(
                f"[{provider.value}/{model}] provider error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _extract_anthropic_text(response: object) -> str:
        blocks = getattr(response, "content", None)
        if not isinstance(blocks, list):
            raise LLMResponseError("Anthropic response had no content blocks")
        parts: list[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise LLMResponseError(
                "Anthropic response was truncated (stop_reason='max_tokens'); "
                "increase llm_max_tokens"
            )
        if not parts:
            raise LLMResponseError("Anthropic response contained no text blocks")
        return "".join(parts)

    @staticmethod
    def _extract_openai_text(response: object) -> str:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise LLMResponseError("OpenAI response had no choices")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise LLMResponseError("OpenAI response choice had no message")
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            raise LLMResponseError("OpenAI response had no text content")
        return content

    async def aclose(self) -> None:
        """Close any SDK clients that were lazily created."""
        for client in (self._anthropic, self._openai):
            if client is None:
                continue
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # noqa: BLE001
                    pass
        self._anthropic = None
        self._openai = None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- filesystem helpers ------------------------------------------------

    def _resolve_workspace(self, root: str | Path | None = None) -> Path:
        target_root = Path(root) if root is not None else self.settings.workspace
        return target_root.resolve()

    def _safe_target(
        self,
        path: str | Path | None,
        root: str | Path | None = None,
    ) -> Path:
        workspace = self._resolve_workspace(root)
        target = workspace if path is None else workspace / path
        try:
            resolved = target.resolve()
        except (OSError, RuntimeError) as exc:
            raise LLMError(f"Cannot resolve path '{path}': {exc}") from exc
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise LLMError(
                f"Path '{path}' escapes workspace root '{workspace}'"
            ) from exc
        return resolved

    async def read_file(
        self,
        path: str | Path,
        root: str | Path | None = None,
    ) -> str:
        target = self._safe_target(path, root)
        try:
            async with aiofiles.open(target, encoding="utf-8") as f:
                return await f.read()
        except FileNotFoundError as exc:
            raise LLMError(f"File not found: {target}") from exc
        except IsADirectoryError as exc:
            raise LLMError(f"Path is a directory, not a file: {target}") from exc
        except UnicodeDecodeError as exc:
            raise LLMError(f"File is not valid UTF-8: {target}") from exc

    async def write_file(
        self,
        path: str | Path,
        content: str,
        root: str | Path | None = None,
    ) -> None:
        target = self._safe_target(path, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        committed = False
        try:
            async with aiofiles.open(tmp_path, "wb") as f:
                await f.write(data)
            os.replace(tmp_path, target)
            committed = True
        finally:
            if not committed:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    async def list_files(
        self,
        path: str | Path | None = None,
        root: str | Path | None = None,
        max_results: int = 10_000,
        max_depth: int = 16,
    ) -> list[str]:
        start = self._safe_target(path, root)
        if not start.is_dir():
            return []
        return await asyncio.to_thread(
            self._list_files_sync, start, max_results, max_depth
        )

    def _list_files_sync(
        self,
        start: Path,
        max_results: int,
        max_depth: int,
    ) -> list[str]:
        start = Path(start).resolve()
        results: list[str] = []
        visited: set[str] = set()
        for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in self._LIST_SKIP]
            try:
                rel = Path(dirpath).relative_to(start)
                depth = 0 if rel == Path(".") else len(rel.parts)
            except ValueError:
                depth = max_depth
            if depth >= max_depth:
                dirnames[:] = []
                continue
            real = os.path.realpath(dirpath)
            if real in visited:
                dirnames[:] = []
                continue
            visited.add(real)
            for name in filenames:
                results.append(os.path.relpath(os.path.join(dirpath, name), start))
                if len(results) >= max_results:
                    return results
        return results
