from __future__ import annotations

import ast
import asyncio
import itertools
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import AsyncAnthropic
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

logger = logging.getLogger(__name__)

#: Fallback for ``LLMClient(settings=None)`` callers, which shadow the module-level default.
DEFAULT_SETTINGS: Settings = settings

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."
DEFAULT_MAX_TOKENS = 4096

# ``llm_max_retries``/``llm_backoff_seconds`` are optional on Settings: fall back to these
# when the field is missing or unusable so the client keeps working.
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0

#: HTTP statuses worth retrying: rate limits, request timeouts and server-side failures.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: Connection/timeout failures carry no status code, so they are matched by type instead.
#: ``TimeoutError`` is spelled out because on py3.10 ``asyncio.TimeoutError`` is a different
#: class, and httpx timeout errors are wrapped by the SDKs' ``APITimeoutError``.
CONNECTION_ERROR_TYPES: tuple[type[BaseException], ...] = (
    AnthropicAPIConnectionError,
    OpenAIAPIConnectionError,
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

#: Directory names that never belong in LLM context (vendored code, caches, build output).
DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        "__pycache__",
        "node_modules",
        "venv",
        "dist",
        "build",
        "site-packages",
        "target",
        "vendor",
        "htmlcov",
    }
)
DEFAULT_MAX_FILES = 200

_MAX_PREVIEW_CHARS = 500
_CODE_FENCE_RE = re.compile(
    r"```[ \t]*[A-Za-z0-9_+.-]*[ \t]*\r?\n(.*?)(?:\r?\n)?[ \t]*```",
    re.DOTALL,
)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_CLOSERS = frozenset("}]")
_OPENERS = {"{": "}", "[": "]"}


def _strip_code_fences(text: str) -> str:
    """Return the body of the first markdown code fence, or ``text`` stripped."""
    match = _CODE_FENCE_RE.search(text)
    if match is None:
        return text.strip()
    return match.group(1).strip()


def _match_region(text: str, start: int) -> int | None:
    """Return the index of the brace/bracket closing the region opened at ``start``.

    String literals and escapes are respected so braces inside JSON strings do not corrupt
    the depth count. ``None`` means the region is never closed.
    """
    stack = [_OPENERS[text[start]]]
    in_string = False
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
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
        elif char in _OPENERS:
            stack.append(_OPENERS[char])
        elif char in _CLOSERS:
            if char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return index
    return None


def _iter_json_regions(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}``/``[...]`` region of ``text`` in order of appearance."""
    index = 0
    while index < len(text):
        if text[index] not in _OPENERS:
            index += 1
            continue
        end = _match_region(text, index)
        if end is None:
            index += 1
            continue
        yield text[index : end + 1]
        index = end + 1


def _strip_trailing_commas(text: str) -> str:
    """Drop commas directly preceding ``}``/``]``, leaving string literals untouched."""
    parts: list[str] = []
    segment_start = 0
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
            parts.append(_TRAILING_COMMA_RE.sub(r"\1", text[segment_start:index]))
            segment_start = index
            in_string = True
    parts.append(_TRAILING_COMMA_RE.sub(r"\1", text[segment_start:]))
    return "".join(parts)


def _literal_container(payload: str) -> dict | list | None:
    """Parse Python-style JSON-ish literals (``True``/``None``/single quotes), if any."""
    try:
        value = ast.literal_eval(payload)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, (dict, list)) else None


def extract_json(text: str) -> Any:
    """Extract the first valid JSON value from LLM output.

    Handles the usual LLM noise: markdown ```` ```json ```` fences, prose before/after the
    payload, trailing commas, and Python-style literals such as ``True``/``None``. Raises
    ``ValueError`` (with a truncated copy of the input) when nothing parses.
    """
    if not isinstance(text, str):
        raise TypeError(f"extract_json() expects str, got {type(text).__name__}")

    stripped = text.strip()
    # Fenced body first: it is the only part of a fenced response that can be valid JSON.
    candidates: list[str] = [stripped]
    fence = _CODE_FENCE_RE.search(text)
    if fence is not None:
        body = fence.group(1).strip()
        if body and body != stripped:
            candidates.insert(0, body)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            pass

    for candidate in candidates:
        for region in _iter_json_regions(candidate):
            for payload in (region, _strip_trailing_commas(region)):
                try:
                    return json.loads(payload)
                except ValueError:
                    pass
                value = _literal_container(payload)
                if value is not None:
                    return value

    preview = stripped if len(stripped) <= _MAX_PREVIEW_CHARS else stripped[:_MAX_PREVIEW_CHARS]
    ellipsis = "..." if len(stripped) > _MAX_PREVIEW_CHARS else ""
    raise ValueError(f"Could not extract JSON from model output: {preview!r}{ellipsis}")


def _require_api_key(value: str | None, env_var: str) -> str:
    key = (value or os.getenv(env_var) or "").strip()
    if not key:
        raise ValueError(f"{env_var} is not set")
    return key


def _attr(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _anthropic_text(response: Any) -> str:
    """Join the text blocks of an Anthropic message, tolerating empty/non-text content."""
    blocks = _attr(response, "content") or []
    return "".join(text for text in (_attr(b, "text") for b in blocks) if isinstance(text, str))


def _openai_text(response: Any) -> str:
    """Read the assistant text of an OpenAI-style completion, tolerating empty content."""
    choices = _attr(response, "choices") or []
    if not choices:
        return ""
    content = _attr(_attr(choices[0], "message"), "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            text for text in (_attr(part, "text") for part in content) if isinstance(text, str)
        )
    return ""


def _is_retryable(exc: BaseException) -> bool:
    """Transient failures only: never retry auth errors (401/403) or cancellation."""
    if isinstance(exc, asyncio.CancelledError):
        return False
    status = _attr(exc, "status_code")
    if not isinstance(status, int):
        status = _attr(exc, "http_status")
    if isinstance(status, int):
        if status in RETRYABLE_STATUS_CODES:
            return True
        if 400 <= status < 500:
            return False
    return isinstance(exc, CONNECTION_ERROR_TYPES)


def _is_excluded_dir(name: str, excluded: set[str]) -> bool:
    return name.startswith(".") or name in excluded


def _iter_files(root: Path, excluded: set[str]) -> Iterator[Path]:
    """Walk ``root`` in sorted order, pruning excluded and hidden directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(n for n in dirnames if not _is_excluded_dir(n, excluded))
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.is_symlink() and not path.exists():
                continue
            yield path


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings: Settings = settings if settings is not None else DEFAULT_SETTINGS
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def max_retries(self) -> int:
        raw = getattr(self.settings, "llm_max_retries", None)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return DEFAULT_MAX_RETRIES

    @property
    def backoff_seconds(self) -> float:
        raw = getattr(self.settings, "llm_backoff_seconds", None)
        try:
            backoff = float(raw)
        except (TypeError, ValueError):
            return DEFAULT_BACKOFF_SECONDS
        return max(0.0, backoff)

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = _require_api_key(self.settings.anthropic_api_key, "ANTHROPIC_API_KEY")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = _require_api_key(self.settings.openai_api_key, "OPENAI_API_KEY")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """Ollama serves the OpenAI chat-completions API, so no new dependency is needed."""
        if self._ollama is None:
            base_url = (self.settings.ollama_base_url or "").strip() or DEFAULT_OLLAMA_BASE_URL
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            self._ollama = AsyncOpenAI(base_url=base_url, api_key="ollama")
        return self._ollama

    @property
    def chat_client(self) -> AsyncOpenAI:
        """The OpenAI-compatible client matching the configured provider."""
        if self.settings.provider == Provider.OLLAMA:
            return self.ollama
        return self.openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._complete_openai(prompt, system, model)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _with_retries(self, operation: str, call: Callable[[], Awaitable[str]]) -> str:
        """Run ``call`` with exponential backoff on transient provider errors.

        ``settings.llm_max_retries`` counts the *extra* attempts after the first one, so the
        default of 3 means up to 4 calls spaced by
        ``settings.llm_backoff_seconds * 2 ** attempt`` seconds. Authentication errors
        (401/403), other 4xx and ``asyncio.CancelledError`` are re-raised immediately.
        """
        max_retries = self.max_retries
        backoff = self.backoff_seconds
        for attempt in range(max_retries + 1):
            try:
                return await call()
            except Exception as exc:
                if attempt >= max_retries or not _is_retryable(exc):
                    raise
                delay = backoff * 2**attempt
                logger.warning(
                    "%s failed with %s (attempt %d/%d); retrying in %.2fs",
                    operation,
                    exc,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"{operation} exhausted retries")  # pragma: no cover - unreachable

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        async def call() -> str:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=system or DEFAULT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return _anthropic_text(response)

        return await self._with_retries("anthropic.messages.create", call)

    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        client: AsyncOpenAI | None = None,
    ) -> str:
        chat_client = client or self.chat_client

        async def call() -> str:
            response = await chat_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return _openai_text(response)

        provider = "ollama" if chat_client is self._ollama else "openai"
        return await self._with_retries(f"{provider}.chat.completions.create", call)

    async def read_file(self, path: str | Path) -> str:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        if target.is_dir():
            raise IsADirectoryError(f"Expected a file but found a directory: {target}")
        async with aiofiles.open(target, encoding="utf-8") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as f:
            await f.write(content)

    def list_files(
        self,
        directory: str | Path,
        *,
        exclude: Iterable[str] | None = None,
        max_files: int | None = DEFAULT_MAX_FILES,
    ) -> list[str]:
        """List files under ``directory`` (relative paths), sorted and free of build noise.

        Hidden directories plus the usual dependency/build directories are pruned, and at
        most ``max_files`` entries come back (``None`` or ``0`` disables the cap). Any name in
        ``exclude`` is pruned in addition to :data:`DEFAULT_EXCLUDED_DIRS`.
        """
        root = Path(directory)
        if not root.is_dir():
            return []
        excluded = set(DEFAULT_EXCLUDED_DIRS)
        excluded.update(str(name).strip("/") for name in exclude or ())
        limit = max_files if max_files and max_files > 0 else None
        files = [
            str(path.relative_to(root))
            for path in itertools.islice(_iter_files(root, excluded), limit)
        ]
        return sorted(files)
