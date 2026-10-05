from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, TypeVar

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

log = structlog.get_logger(__name__)

T = TypeVar("T")

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

DEFAULT_HTTP_TIMEOUT = 120.0
MAX_BACKOFF_SECONDS = 60.0
ERROR_SNIPPET_LIMIT = 300
CLOSE_CANCEL_RETRIES = 3
"""How often a close re-awaits its shielded task when a cancellation keeps landing."""

SKIP_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".cache",
        ".eggs",
        ".idea",
        ".mypy_cache",
        ".next",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".vscode",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)
SKIP_DIRECTORY_SUFFIXES: tuple[str, ...] = (".egg-info",)
SKIP_FILE_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo")

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 425, 429, 529})
NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset(
    {400, 401, 402, 403, 404, 405, 406, 410, 413, 415, 422}
)
NON_RETRYABLE_EXCEPTION_NAMES: frozenset[str] = frozenset(
    {
        "AuthenticationError",
        "BadRequestError",
        "ConflictError",
        "InvalidRequestError",
        "NotFoundError",
        "PermissionDeniedError",
        "UnprocessableEntityError",
    }
)
RETRYABLE_NAME_TOKENS: tuple[str, ...] = (
    "apiconnection",
    "apitimeout",
    "connection",
    "internalserver",
    "overloaded",
    "ratelimit",
    "serviceunavailable",
    "timeout",
)
RETRYABLE_MESSAGE_TOKENS: tuple[str, ...] = (
    "connection reset",
    "connection aborted",
    "overloaded",
    "rate limit",
    "service unavailable",
    "server had an error",
    "temporarily unavailable",
    "timed out",
    "too many requests",
    "try again",
)

_FENCE_PATTERN = re.compile(r"```[a-zA-Z0-9_+.-]*[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read *key* from a mapping or an SDK model object."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _truncate(text: str, limit: int = ERROR_SNIPPET_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (+{len(text) - limit} chars)"


def exception_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status code for an exception raised by an SDK or httpx."""
    for attribute in ("status_code", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def is_retryable_error(exc: BaseException) -> bool:
    """True when *exc* is a transient provider failure worth retrying.

    Rate limits, connection/timeout errors, 5xx responses and "overloaded"
    responses are retried. Authentication (401/403) and invalid request (400)
    errors are terminal and fail fast.
    """
    status = exception_status_code(exc)
    if status is not None:
        if status in NON_RETRYABLE_STATUS_CODES:
            return False
        return status in RETRYABLE_STATUS_CODES or status >= 500
    if isinstance(exc, (httpx.TransportError, httpx.RemoteProtocolError, asyncio.TimeoutError)):
        return True
    name = type(exc).__name__
    if name in NON_RETRYABLE_EXCEPTION_NAMES:
        return False
    lowered = name.lower()
    if any(token in lowered for token in RETRYABLE_NAME_TOKENS):
        return True
    message = str(exc).lower()
    return any(token in message for token in RETRYABLE_MESSAGE_TOKENS)


def _skip_directory(name: str) -> bool:
    return name in SKIP_DIRECTORIES or name.endswith(SKIP_DIRECTORY_SUFFIXES)


def _iter_json_objects(text: str) -> Iterator[str]:
    """Yield every top-level ``{...}`` block in *text*, outermost first."""
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : index + 1]
                start = -1
    if depth > 0 and start >= 0:
        yield text[start:]


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse a single JSON object out of raw LLM output.

    Handles bare JSON, ```json fenced blocks and JSON wrapped in prose. Raises
    :class:`ValueError` with a truncated snippet of *raw* when nothing parses.
    """
    if not isinstance(raw, str):
        raise TypeError(f"extract_json expects str, got {type(raw).__name__}")
    text = raw.strip()
    snippet = _truncate(text)
    if not text:
        raise ValueError(f"Cannot extract JSON from an empty response. Snippet: {snippet!r}")

    candidates: list[str] = [text]
    candidates.extend(match.group(1).strip() for match in _FENCE_PATTERN.finditer(text))
    candidates.extend(_iter_json_objects(text))

    problems: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            problems.append(str(exc))
            continue
        if isinstance(parsed, dict):
            return parsed
        problems.append(f"top level JSON was a {type(parsed).__name__}, not an object")
    raise ValueError(
        "Could not extract a JSON object from the LLM response "
        f"({'; '.join(problems[:3]) or 'no JSON object found'}). Snippet: {snippet!r}"
    )


async def _aclose_client(client: httpx.AsyncClient) -> None:
    """Close *client*, completing the close even if the caller is cancelled.

    ``LLMClient.aclose()`` is awaited from ``finally`` blocks that may already be
    unwinding because of :class:`asyncio.CancelledError` (the CLI and the web
    layer both do this when a run is cancelled or a client disconnects).  An
    ``await`` on httpx's ``aclose()`` raises immediately in that state, so the
    client is never closed and its sockets leak.

    Running the close in its own task and awaiting it through
    :func:`asyncio.shield` fixes that: a cancellation aimed at the caller tears
    down only the shield, never the in-flight close, and the loop re-awaits the
    shielded task so the sockets are really released before returning.
    """
    close = asyncio.ensure_future(client.aclose())
    for attempt in range(1, CLOSE_CANCEL_RETRIES + 1):
        try:
            await asyncio.shield(close)
        except asyncio.CancelledError:
            if close.done():
                return
            log.warning("llm_aclose_cancelled_during_close", attempt=attempt)
        except Exception as exc:  # noqa: BLE001 - a failed close must not raise into cleanup
            log.warning("llm_aclose_failed", error=_truncate(str(exc)))
            return
        else:
            return
    if not close.done():
        log.warning("llm_aclose_detached", retries=CLOSE_CANCEL_RETRIES)


class LLMClient:
    """Thin async wrapper around the supported LLM providers.

    Passing ``settings=None`` (the default) reuses the module level
    :data:`furrow.config.settings` singleton, which keeps
    ``LLMClient(settings=settings)`` working when the caller has no settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http: httpx.AsyncClient | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set; export it or set FURROW_ANTHROPIC_API_KEY."
                )
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set; export it or set FURROW_OPENAI_API_KEY."
                )
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT)
        return self._http

    @property
    def ollama_chat_url(self) -> str:
        return f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"

    async def aclose(self) -> None:
        """Release the lazily created HTTP client, if any.

        Cancellation safe and idempotent: the close runs through
        :func:`_aclose_client` so a ``finally`` block that is already unwinding
        from ``CancelledError`` still releases the client, the reference is
        dropped before awaiting so a caller can never receive a client that is
        mid-close, and a second call (or one after a failed close) is a no-op.
        """
        client, self._http = self._http, None
        if client is None:
            return
        await _aclose_client(client)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = getattr(self.settings.provider, "value", self.settings.provider)
        log.info(
            "llm_request",
            provider=provider,
            model=model,
            prompt_chars=len(prompt),
            system_chars=len(system or ""),
        )
        if self.settings.provider == Provider.ANTHROPIC:
            text = await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            text = await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            text = await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider!r}")
        log.debug("llm_response", provider=provider, model=model, response_chars=len(text))
        return text

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        system_prompt = system or DEFAULT_SYSTEM_PROMPT

        async def call() -> Any:
            return await self.anthropic.messages.create(
                model=model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

        response = await self._with_retries(f"anthropic:{model}", call)
        return self._anthropic_text(response, model)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        system_prompt = system or DEFAULT_SYSTEM_PROMPT

        async def call() -> Any:
            return await self.openai.chat.completions.create(
                model=model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

        response = await self._with_retries(f"openai:{model}", call)
        choices = _get(response, "choices") or []
        if not choices:
            raise ValueError(
                f"OpenAI returned an empty `choices` list for model {model!r}. "
                "The response usually means the model produced no output; "
                "check the prompt, the model id and your API quota."
            )
        message = _get(choices[0], "message")
        content = _get(message, "content") if message is not None else None
        if not isinstance(content, str):
            log.warning("llm_missing_content", model=model, provider="openai")
            return ""
        return content

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = self.ollama_chat_url
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.settings.llm_temperature,
                "num_predict": self.settings.llm_max_tokens,
            },
        }

        async def call() -> httpx.Response:
            response = await self.http.post(url, json=payload)
            if response.status_code != 200:
                raise httpx.HTTPStatusError(
                    f"Ollama returned HTTP {response.status_code} for POST {url}: "
                    f"{_truncate(response.text)}",
                    request=response.request,
                    response=response,
                )
            return response

        response = await self._with_retries(f"ollama:{model}", call)
        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError(
                f"Ollama returned a non-JSON body from {url}: {_truncate(response.text)}"
            ) from exc
        message = data.get("message") if isinstance(data, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Ollama returned no message content from {url} for model {model!r}. "
                "Verify the model is pulled locally (`ollama list`) and reachable."
            )
        return content

    async def _with_retries(self, operation: str, call: Callable[[], Awaitable[T]]) -> T:
        attempts = max(1, int(self.settings.retry_attempts))
        base = max(0.0, float(self.settings.retry_min_wait))
        retrying = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=base, min=base, max=max(base, MAX_BACKOFF_SECONDS)),
            retry=retry_if_exception(is_retryable_error),
            reraise=True,
        )
        attempt_number = 0
        async for attempt_state in retrying:
            attempt_number += 1
            with attempt_state:
                try:
                    result = await call()
                except Exception as exc:
                    self._log_failure(operation, attempt_number, attempts, exc)
                    raise
                log.debug("llm_call_succeeded", operation=operation, attempt=attempt_number)
                return result
        raise RuntimeError(f"{operation}: retry loop ended without returning a result")

    def _log_failure(
        self, operation: str, attempt_number: int, attempts: int, exc: BaseException
    ) -> None:
        details = {
            "operation": operation,
            "attempt": attempt_number,
            "max_attempts": attempts,
            "error_type": type(exc).__name__,
            "status": exception_status_code(exc),
            "error": _truncate(str(exc)),
        }
        if not is_retryable_error(exc):
            log.error("llm_request_failed", **details)
        elif attempt_number < attempts:
            log.warning("llm_retry", **details)
        else:
            log.error("llm_retries_exhausted", **details)

    @staticmethod
    def _anthropic_text(response: Any, model: str) -> str:
        """Join every text block of an Anthropic response.

        Non-text blocks (``thinking``, ``tool_use``, ...) are ignored instead of
        raising ``AttributeError`` as ``response.content[0].text`` would.
        """
        parts: list[str] = []
        block_types: list[str] = []
        for block in _get(response, "content") or []:
            block_type = _get(block, "type")
            text = _get(block, "text")
            block_types.append(str(block_type))
            if isinstance(text, str) and text and block_type in (None, "text"):
                parts.append(text)
        if not parts:
            raise ValueError(
                f"Anthropic returned no text content for model {model!r} "
                f"(content block types: {block_types or ['<empty>']}). "
                "The model may have returned only thinking or tool_use blocks; "
                "enable streaming text or send a prompt that does not force tool use."
            )
        return "".join(parts)

    def extract_json(self, raw: str) -> dict[str, Any]:
        """Extract a JSON object from raw LLM output (see extract_json_object)."""
        return extract_json_object(raw)

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, encoding="utf-8") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w", encoding="utf-8") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """Return file paths relative to *directory*, skipping build/VCS noise."""
        root = Path(directory)
        if not root.is_dir():
            return []
        results: list[str] = []
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not _skip_directory(d))
            current_path = Path(current)
            for name in sorted(filenames):
                if name.endswith(SKIP_FILE_SUFFIXES):
                    continue
                results.append((current_path / name).relative_to(root).as_posix())
        return results
