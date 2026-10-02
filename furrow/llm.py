from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

# --- module level tunables -------------------------------------------------

DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

#: HTTP statuses worth retrying: request timeout, lock conflict, rate limit and
#: transient server side failures.  4xx auth/validation errors (400/401/403/422)
#: are deliberately absent - retrying them just burns quota.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})

RETRY_ATTEMPTS = 3
RETRY_WAIT_MULTIPLIER = 1.0
RETRY_WAIT_MIN = 1.0
RETRY_WAIT_MAX = 10.0

IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".idea",
        "dist",
        "build",
    }
)
IGNORED_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo")

_MAX_ERROR_TEXT_CHARS = 2000

# ```json ... ``` / ```JSON ... ``` / ``` ... ``` (language tag optional).
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n?(.*?)```", re.DOTALL)

_OPEN_TO_CLOSE = {"{": "}", "[": "]"}


class TruncatedResponseError(RuntimeError):
    """Raised when an LLM response was cut off at the token limit."""


# --- JSON extraction -------------------------------------------------------


def _fenced_bodies(text: str) -> list[str]:
    """Return the bodies of every fenced code block in *text* (may be empty)."""
    return [body for _tag, body in _FENCE_RE.findall(text) if body.strip()]


def _balanced_candidates(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}`` / ``[...]`` region found in *text*.

    Braces and brackets inside JSON string literals are ignored, and backslash
    escapes are honoured, so ``{"a": "}{"}`` yields a single candidate.
    """
    length = len(text)
    for start, char in enumerate(text):
        closer = _OPEN_TO_CLOSE.get(char)
        if closer is None:
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, length):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == char:
                depth += 1
            elif current == closer:
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def extract_json(text: str) -> Any:
    """Extract the first JSON value from LLM output.

    Tolerates markdown code fences (```json, ```JSON or a bare ```) and prose
    wrapped around the payload.  Returns the decoded value, or raises
    ``ValueError`` quoting the offending text when nothing parses.
    """
    text = text if isinstance(text, str) else str(text)

    last_error: Exception | None = None
    for source in [*_fenced_bodies(text), text]:
        for candidate in _balanced_candidates(source):
            try:
                return json.loads(candidate)
            except ValueError as exc:  # json.JSONDecodeError subclasses ValueError
                last_error = exc

    snippet = text if len(text) <= _MAX_ERROR_TEXT_CHARS else text[:_MAX_ERROR_TEXT_CHARS]
    if last_error is None:
        reason = "no JSON object or array was found in the output"
    else:
        reason = f"last parse error was: {last_error}"
    raise ValueError(f"Could not extract JSON from LLM output: {snippet!r} ({reason})")


# --- retry helpers ---------------------------------------------------------


def _status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_transient_error(exc: BaseException) -> bool:
    """True for errors that are worth retrying (network blips, 5xx, rate limits)."""
    if isinstance(
        exc,
        (
            httpx.TransportError,
            httpx.TimeoutException,
            anthropic.APIConnectionError,
            openai.APIConnectionError,
            asyncio.TimeoutError,
            TimeoutError,
        ),
    ):
        return True
    if _status_code(exc) is not None:
        return _status_code(exc) in RETRYABLE_STATUS_CODES
    return False


# --- client ----------------------------------------------------------------


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    # -- configuration helpers ------------------------------------------------

    @property
    def request_timeout(self) -> float:
        """Per request timeout, honouring an optional ``request_timeout`` setting."""
        value = getattr(self.settings, "request_timeout", None)
        if value is None:
            return DEFAULT_REQUEST_TIMEOUT_SECONDS
        try:
            timeout = float(value)
        except (TypeError, ValueError):
            return DEFAULT_REQUEST_TIMEOUT_SECONDS
        return timeout if timeout > 0 else DEFAULT_REQUEST_TIMEOUT_SECONDS

    # -- SDK clients ----------------------------------------------------------

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key, timeout=self.request_timeout)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key, timeout=self.request_timeout)
        return self._openai

    # -- retry ----------------------------------------------------------------

    async def _call_with_retry(self, operation: Callable[[], Awaitable[str]]) -> str:
        """Await *operation*, retrying transient failures with exponential backoff."""
        retrying = AsyncRetrying(
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            wait=wait_exponential(
                multiplier=RETRY_WAIT_MULTIPLIER,
                min=RETRY_WAIT_MIN,
                max=RETRY_WAIT_MAX,
            ),
            retry=retry_if_exception(is_transient_error),
            reraise=True,
        )
        async for attempt in retrying:  # type: ignore[attr-defined]
            with attempt:
                return await operation()
        raise RuntimeError("retry loop finished without producing a result")

    # -- completion -----------------------------------------------------------

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        if provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        max_tokens = getattr(self.settings, "max_tokens", None) or DEFAULT_MAX_TOKENS

        async def call() -> str:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system or DEFAULT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            if str(getattr(response, "stop_reason", "") or "") == "max_tokens":
                raise TruncatedResponseError(
                    f"Anthropic response from model {model!r} was truncated at the "
                    f"max_tokens limit of {max_tokens}; the reply is incomplete. "
                    "Raise max_tokens or split the request."
                )
            return response.content[0].text

        return await self._call_with_retry(call)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        async def call() -> str:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            choice = response.choices[0]
            if str(getattr(choice, "finish_reason", "") or "") == "length":
                raise TruncatedResponseError(
                    f"OpenAI response from model {model!r} was truncated at the "
                    "token limit; the reply is incomplete. Split the request."
                )
            return choice.message.content or ""

        return await self._call_with_retry(call)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }
        timeout = self.request_timeout

        async def call() -> str:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
            if not isinstance(data, dict) or "response" not in data:
                raise ValueError(
                    f"Ollama response from {url} is missing the 'response' field: {data!r}"
                )
            return str(data["response"])

        return await self._call_with_retry(call)

    # -- filesystem -----------------------------------------------------------

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """List workspace files relative to *directory*, pruning noisy trees.

        ``dirnames`` is pruned in place so ``.git``/``node_modules``/``venv``
        and friends are never descended into.  Results are sorted for
        determinism.
        """
        root = Path(directory)
        if not root.exists():
            return []

        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
            for filename in filenames:
                if filename.endswith(IGNORED_SUFFIXES):
                    continue
                absolute = os.path.join(dirpath, filename)
                results.append(str(Path(os.path.relpath(absolute, root))))
        return sorted(results)
