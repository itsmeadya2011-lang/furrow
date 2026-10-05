"""Provider-agnostic async LLM client used by the furrow agents.

The client keeps one lazily built client per provider (Anthropic SDK, OpenAI
SDK or a plain :mod:`httpx` client for Ollama), applies a shared retry policy
and a per-request timeout to every completion, and exposes the small
filesystem helpers the worker agent relies on.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
from typing import Any, Final

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    wait_exponential,
)
from tenacity.stop import stop_base

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

DEFAULT_SYSTEM_PROMPT: Final[str] = "You are a helpful coding assistant."

# HTTP status codes worth retrying; anything 5xx is retried as well.
_TRANSIENT_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})

# Longest provider error body embedded in an exception message.
_ERROR_BODY_LIMIT: Final[int] = 500


class MissingAPIKeyError(ValueError):
    """Raised when no API key can be resolved for a provider."""


class LLMResponseError(ValueError):
    """Raised when a provider reply contains no usable text."""


def _transient_exception_types() -> tuple[type[BaseException], ...]:
    """Collect transient error classes from the installed SDKs.

    The modules are resolved through :func:`importlib.import_module` and the
    symbols through :func:`getattr`, so a renamed or missing optional symbol can
    never break importing this module; the tuple only feeds the retry predicate.
    """
    collected: list[type[BaseException]] = [
        httpx.TransportError,
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    ]
    for module_name in ("anthropic", "openai"):
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - both SDKs are hard dependencies
            continue
        for symbol in ("APIConnectionError", "APITimeoutError", "RateLimitError"):
            candidate = getattr(module, symbol, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                collected.append(candidate)
    return tuple(collected)


def _has_transient_status(exception: BaseException) -> bool:
    """Return ``True`` when a provider error carries a retryable HTTP status."""
    status = getattr(exception, "status_code", None)
    if status is None:
        status = getattr(getattr(exception, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS_CODES or status >= 500
    return False


def _body_snippet(response: httpx.Response) -> str:
    """Return a bounded, single-line excerpt of a response body for error messages."""
    body = " ".join(response.text.split())
    if len(body) > _ERROR_BODY_LIMIT:
        return body[:_ERROR_BODY_LIMIT] + "..."
    return body or "<empty body>"


_RETRYABLE = retry_if_exception_type(_transient_exception_types()) | retry_if_exception(
    _has_transient_status
)


class _StopAfterClientRetries(stop_base):
    """Stop once the bound client's ``llm_max_retries`` attempt budget is spent.

    tenacity evaluates ``stop`` once, when the decorator is applied, so a
    per-instance budget can only be read from the retry state (which carries the
    call arguments, i.e. ``self`` for a bound method). Clients without a usable
    setting fall back to the module-level default.
    """

    def __call__(self, retry_state: RetryCallState) -> bool:
        client = retry_state.args[0] if retry_state.args else None
        attempts = getattr(getattr(client, "settings", None), "llm_max_retries", None)
        if not isinstance(attempts, int):
            attempts = default_settings.llm_max_retries
        return retry_state.attempt_number >= max(1, attempts)


class LLMClient:
    """Async facade over the supported LLM providers plus small file helpers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: httpx.AsyncClient | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazily built Anthropic client, keyed off the settings then the env var."""
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise MissingAPIKeyError(
                    "ANTHROPIC_API_KEY is not set. Provide the key as the "
                    "FURROW_ANTHROPIC_API_KEY setting or as the ANTHROPIC_API_KEY "
                    "environment variable before using the anthropic provider."
                )
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazily built OpenAI client, keyed off the settings then the env var."""
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise MissingAPIKeyError(
                    "OPENAI_API_KEY is not set. Provide the key as the "
                    "FURROW_OPENAI_API_KEY setting or as the OPENAI_API_KEY "
                    "environment variable before using the openai provider."
                )
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> httpx.AsyncClient:
        """Lazily built HTTP client aimed at ``settings.ollama_base_url``."""
        if self._ollama is None:
            self._ollama = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url.rstrip("/"),
                timeout=httpx.Timeout(self.settings.llm_timeout_seconds),
            )
        return self._ollama

    @retry(
        retry=_RETRYABLE,
        stop=_StopAfterClientRetries(),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Return one completion for ``prompt``, retrying transient failures."""
        provider = self.settings.provider
        if provider == Provider.OLLAMA:
            model = model or self.settings.ollama_model
        else:
            model = model or self.settings.model
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {provider!r}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.settings.llm_timeout_seconds,
        )
        blocks = list(getattr(response, "content", None) or [])
        stop_reason = getattr(response, "stop_reason", None)
        if not blocks:
            raise LLMResponseError(
                f"Anthropic returned no content blocks for model {model!r} "
                f"(stop_reason={stop_reason!r})"
            )
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text
        block_types = [getattr(block, "type", type(block).__name__) for block in blocks]
        raise LLMResponseError(
            f"Anthropic returned {len(blocks)} content block(s) without text for "
            f"model {model!r} (stop_reason={stop_reason!r}, types={block_types!r})"
        )

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            timeout=self.settings.llm_timeout_seconds,
        )
        choices = list(getattr(response, "choices", None) or [])
        if not choices:
            raise LLMResponseError(f"OpenAI returned no choices for model {model!r}")
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            finish_reason = getattr(choice, "finish_reason", None)
            raise LLMResponseError(
                f"OpenAI returned an empty message for model {model!r} "
                f"(finish_reason={finish_reason!r}, "
                f"message_type={type(message).__name__})"
            )
        return content

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"num_predict": self.settings.max_tokens},
        }
        response = await self.ollama.post("/api/chat", json=payload)
        if response.is_error:
            raise httpx.HTTPStatusError(
                f"Ollama returned HTTP {response.status_code} for model {model!r} "
                f"at {response.request.url}: {_body_snippet(response)}",
                request=response.request,
                response=response,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                f"Ollama returned a non-JSON body for model {model!r}: "
                f"{_body_snippet(response)!r}"
            ) from exc
        if not isinstance(data, dict):
            raise LLMResponseError(
                f"Ollama returned a JSON {type(data).__name__} payload for model "
                f"{model!r}; expected an object"
            )
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError(
                f"Ollama returned an empty message for model {model!r} "
                f"(done_reason={data.get('done_reason')!r}, keys={sorted(data)!r})"
            )
        return content

    async def aclose(self) -> None:
        """Close the lazily created Ollama HTTP client, if one was ever built."""
        if self._ollama is not None:
            await self._ollama.aclose()
            self._ollama = None

    async def read_file(self, path: str | Path) -> str:
        """Read ``path`` without blocking the event loop."""
        async with aiofiles.open(path, encoding="utf-8") as handle:
            return await handle.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write ``content`` to ``path``, creating parent directories off-loop."""
        target = Path(path)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as handle:
            await handle.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """Return every file under ``directory`` as sorted relative paths.

        This walk is synchronous and touches the filesystem for every entry, so
        call :meth:`list_files_async` from async code (event loop) and keep this
        method for synchronous callers and small trees only.
        """
        root = Path(directory)
        if not root.exists():
            return []
        found = (str(entry.relative_to(root)) for entry in root.rglob("*") if entry.is_file())
        return sorted(found)

    async def list_files_async(self, directory: str | Path) -> list[str]:
        """Async wrapper around :meth:`list_files` that keeps the walk off the loop."""
        return await asyncio.to_thread(self.list_files, directory)