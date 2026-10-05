from __future__ import annotations

import asyncio
import inspect
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
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

log = structlog.get_logger("furrow.llm")

DEFAULT_SYSTEM = "You are a helpful coding assistant."
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

_PRUNED_DIRECTORIES = frozenset(
    {
        ".furrow",
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

# Failures worth another attempt: provider SDK, transport and timeout errors.
# A missing API key raises ValueError and is deliberately absent here.
_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.RateLimitError,
    httpx.TransportError,
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
)


class LLMResponseError(RuntimeError):
    """Raised when a provider replies successfully but carries no usable text."""


def _default_settings() -> Settings:
    """Return the process-wide settings singleton defined in :mod:`furrow.config`."""
    return default_settings


def _provider_name(provider: Any) -> str:
    """Return a provider as a string, tolerating raw string configuration values."""
    return str(getattr(provider, "value", provider))


def _as_float(value: Any, default: float) -> float:
    """Coerce a settings value to a positive float, falling back to ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _running_loop_exists() -> bool:
    """Return True when an event loop is already running in this thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


async def _close_quietly(client: Any, label: str) -> None:
    """Close a provider client, tolerating already-closed clients and failures."""
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - cleanup must never mask the caller's error
        log.debug("llm.client_close_failed", client=label, error=type(exc).__name__)


def _anthropic_text(response: Any) -> str:
    """Concatenate every text block of an Anthropic response.

    Non-text blocks (``thinking``, ``tool_use``, ...) are skipped, so an empty or
    reasoning-only response raises :class:`LLMResponseError` instead of failing with
    ``IndexError`` or ``AttributeError`` on ``content[0]``.
    """
    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    if not parts:
        raise LLMResponseError("Anthropic response contained no text blocks")
    return "".join(parts)


class LLMClient:
    """Thin async wrapper around the supported chat completion providers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else _default_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ config

    @property
    def timeout_seconds(self) -> float:
        """Per-attempt timeout in seconds (``llm_timeout_seconds``, default 120)."""
        return _as_float(
            getattr(self.settings, "llm_timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            DEFAULT_TIMEOUT_SECONDS,
        )

    @property
    def max_attempts(self) -> int:
        """Attempts allowed per completion (``llm_max_retries``, default 3, minimum 1)."""
        configured = getattr(self.settings, "llm_max_retries", DEFAULT_MAX_ATTEMPTS)
        try:
            attempts = int(configured)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return DEFAULT_MAX_ATTEMPTS
        return max(1, attempts)

    @property
    def max_tokens(self) -> int:
        """Anthropic output token budget (``llm_max_tokens``, default 4096)."""
        value = _as_float(
            getattr(self.settings, "llm_max_tokens", DEFAULT_MAX_TOKENS),
            float(DEFAULT_MAX_TOKENS),
        )
        return int(value)

    # ---------------------------------------------------------------- providers

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazily built ``AsyncAnthropic`` client; requires an API key."""
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set (FURROW_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY)"
                )
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazily built ``AsyncOpenAI`` client; requires an API key."""
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set (FURROW_OPENAI_API_KEY or OPENAI_API_KEY)"
                )
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    def _http_client(self) -> httpx.AsyncClient:
        """Return the lazily created httpx client used for Ollama requests."""
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._httpx

    # -------------------------------------------------------------- completion

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Run one chat completion, bounded by a timeout and retried on transient errors.

        Args:
            prompt: User message content.
            system: Optional system message; a generic assistant prompt is used when empty.
            model: Optional model override; the configured model is used when empty.

        Returns:
            The assistant's text response.

        Raises:
            ValueError: Unsupported provider or missing provider API key (never retried).
            LLMResponseError: The provider replied without usable text.
        """
        provider = self.settings.provider
        resolved_model = model or self.settings.model
        attempt = retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(_TRANSIENT_ERRORS),
            before_sleep=lambda state: self._log_retry(
                state, provider=provider, model=resolved_model
            ),
            reraise=True,
        )(self._attempt)
        return await attempt(provider, prompt, system, resolved_model)

    async def _attempt(self, provider: Provider, prompt: str, system: str, model: str) -> str:
        """Perform a single completion attempt bounded by the configured timeout."""
        provider_name = _provider_name(provider)
        timeout = self.timeout_seconds
        log.debug(
            "llm.completion_started",
            provider=provider_name,
            model=model,
            prompt_chars=len(prompt),
            timeout_seconds=timeout,
        )
        text = await asyncio.wait_for(
            self._call_provider(provider, prompt, system, model), timeout=timeout
        )
        log.debug(
            "llm.completion_finished",
            provider=provider_name,
            model=model,
            response_chars=len(text),
        )
        return text

    async def _call_provider(self, provider: Provider, prompt: str, system: str, model: str) -> str:
        """Dispatch a completion to the concrete provider implementation."""
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        if provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        raise ValueError(f"Unsupported provider: {_provider_name(provider)}")

    def _log_retry(self, retry_state: RetryCallState, *, provider: Any, model: str) -> None:
        """Log a warning before tenacity sleeps prior to the next attempt."""
        outcome = retry_state.outcome
        error = outcome.exception() if outcome is not None else None
        log.warning(
            "llm.completion_retry",
            provider=_provider_name(provider),
            model=model,
            attempt=retry_state.attempt_number,
            max_attempts=self.max_attempts,
            error=type(error).__name__ if error is not None else None,
        )

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        """Complete a prompt through the Anthropic Messages API."""
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            system=system or DEFAULT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return _anthropic_text(response)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        """Complete a prompt through the OpenAI chat completions API."""
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMResponseError("OpenAI response contained no choices")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not content:
            raise LLMResponseError("OpenAI response contained no message content")
        return content

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Complete a prompt through a local Ollama server (non-streaming ``/api/chat``)."""
        base_url = str(getattr(self.settings, "ollama_base_url", DEFAULT_OLLAMA_BASE_URL))
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = await self._http_client().post(f"{base_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content:
            raise LLMResponseError("Ollama response did not contain message content")
        return content

    # --------------------------------------------------------------- lifecycle

    async def aclose(self) -> None:
        """Close every provider client that was constructed. Safe to call repeatedly."""
        clients: list[tuple[str, Any]] = [
            ("httpx", self._httpx),
            ("anthropic", self._anthropic),
            ("openai", self._openai),
        ]
        self._httpx = None
        self._anthropic = None
        self._openai = None
        for label, client in clients:
            if client is not None:
                await _close_quietly(client, label)

    async def __aenter__(self) -> LLMClient:  # noqa: PYI034 - Self needs typing_extensions on 3.10
        """Enter an async context, closing the clients on exit."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Leave the async context by closing every client."""
        await self.aclose()

    def close_sync(self) -> None:
        """Best-effort synchronous cleanup for callers without a running event loop.

        The CLI calls this after ``asyncio.run`` returns, when awaiting is impossible.
        It is a no-op when a loop is already running in this thread, because there the
        caller must await :meth:`aclose` from inside the loop instead.
        """
        if _running_loop_exists():
            log.debug("llm.close_sync_skipped", reason="event loop already running")
            return
        try:
            asyncio.run(self.aclose())
        except RuntimeError as exc:  # loop creation failed; a teardown path must not raise
            log.debug("llm.close_sync_failed", error=type(exc).__name__)

    # ------------------------------------------------------------------- files

    async def read_file(self, path: str | Path) -> str:
        """Read a UTF-8 text file and return its contents."""
        async with aiofiles.open(path, "r", encoding="utf-8") as handle:
            return await handle.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write UTF-8 text to ``path``, creating parent directories as needed."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as handle:
            await handle.write(content)

    async def list_files(self, directory: str | Path | None = None, limit: int = 2000) -> list[str]:
        """List files below ``directory`` as POSIX-style relative paths.

        Noisy directories (``.git``, ``node_modules``, ...) are pruned and at most
        ``limit`` entries are returned. Defaults to the configured workspace.
        """
        if limit <= 0:
            return []
        root = directory if directory is not None else getattr(self.settings, "workspace", ".")
        return await asyncio.to_thread(self._list_files, Path(root), limit)

    def _list_files(self, root: Path, limit: int) -> list[str]:
        """Blocking implementation of :meth:`list_files`, run off the event loop."""
        base = root.resolve()
        if not base.is_dir():
            return []
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(name for name in dirnames if name not in _PRUNED_DIRECTORIES)
            current = Path(dirpath)
            for name in sorted(filenames):
                found.append((current / name).relative_to(base).as_posix())
                if len(found) >= limit:
                    return found
        return found
