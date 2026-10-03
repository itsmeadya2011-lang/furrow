from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

if TYPE_CHECKING:
    # ``typing.Self`` only exists from 3.11 onwards.  On 3.10 fall back to the
    # backport shipped as a transitive dependency of pydantic.  This block is
    # never executed at runtime, so it adds no hard import-time dependency.
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

__all__ = ["LLMClient", "extract_json"]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

# Retries are deliberately small: a transient provider hiccup should not abort a
# whole cycle, but we must not hammer the provider or stall the loop either.
RETRY_ATTEMPTS = 3
RETRY_MAX_DELAY_SECONDS = 30

# Only genuine transient failures are retried. ``ValueError`` (missing API key,
# unusable settings, empty response) and the non-retryable 4xx families
# (BadRequest, AuthenticationError, PermissionDeniedError, NotFoundError, ...)
# are intentionally absent.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    TimeoutError,
    ConnectionError,
)

RETRY_KWARGS: dict[str, Any] = {
    "stop": stop_after_attempt(RETRY_ATTEMPTS) | stop_after_delay(RETRY_MAX_DELAY_SECONDS),
    "wait": wait_exponential(multiplier=1, min=1, max=8),
    "retry": retry_if_exception_type(TRANSIENT_ERRORS),
    "reraise": True,
}

# Directory names that never carry useful signal for an LLM prompt.
IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "dist",
        "build",
        ".next",
        "target",
        ".idea",
        ".vscode",
    }
)
IGNORED_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".swp")
IGNORED_FILE_NAMES: frozenset[str] = frozenset({".DS_Store"})

_CODE_BLOCK_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n")
_BRACKET_PAIRS = {"{": "}", "[": "]"}


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        # Agents hold an optional ``Settings`` and hand it straight through, so
        # ``None`` means "use the module level singleton" rather than an error.
        self.settings = settings if settings is not None else default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    # ---------------------------------------------------------------- clients
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

    @property
    def ollama(self) -> AsyncOpenAI:
        """Ollama exposes an OpenAI-compatible API under ``{base_url}/v1``."""
        if self._ollama is None:
            base_url = self._ollama_base_url
            self._ollama = AsyncOpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        return self._ollama

    @property
    def _ollama_base_url(self) -> str:
        base_url = (self.settings.ollama_base_url or "").strip().rstrip("/")
        # Users frequently paste the full endpoint including /v1; strip it so the
        # suffix added in ``ollama`` can never turn into /v1/v1.
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")].rstrip("/")
        if not base_url:
            raise ValueError("ollama_base_url is not set")
        return base_url

    # ------------------------------------------------------------ completion
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        handlers = {
            Provider.ANTHROPIC: self._complete_anthropic,
            Provider.OPENAI: self._complete_openai,
            Provider.OLLAMA: self._complete_ollama,
        }
        handler = handlers.get(self.settings.provider)
        if handler is None:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")
        return await handler(prompt, system, model)

    @retry(**RETRY_KWARGS)
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return _anthropic_text(response)

    @retry(**RETRY_KWARGS)
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=_chat_messages(prompt, system),
        )
        return _openai_text(response)

    @retry(**RETRY_KWARGS)
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=_chat_messages(prompt, system),
        )
        return _openai_text(response)

    # ------------------------------------------------------------- lifecycle
    async def close(self) -> None:
        """Close every client that was actually created. Best effort."""
        for client in (self._anthropic, self._openai, self._ollama):
            if client is None:
                continue
            try:
                close = getattr(client, "close", None) or getattr(client, "aclose", None)
                if close is None:
                    continue
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:  # pragma: no cover - cleanup must never raise
                logger.debug("Failed to close LLM client", exc_info=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    # ------------------------------------------------------------------ i/o
    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(
        self,
        directory: str | Path,
        ignore: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """List project files relative to ``directory``.

        Junk directories (``.git``, ``__pycache__``, virtualenvs, caches, ...)
        are pruned while walking, so they are never traversed into. ``ignore``
        adds extra names to the skip list and ``limit`` caps the number of
        returned paths (``None`` means unlimited).
        """
        root = Path(directory)
        if not root.is_dir():
            return []
        ignored = IGNORED_DIRECTORIES | {str(name) for name in (ignore or ())}
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Pruning in place stops os.walk from descending into these at all.
            dirnames[:] = sorted(name for name in dirnames if name not in ignored)
            for name in sorted(filenames):
                if name in IGNORED_FILE_NAMES or name.endswith(IGNORED_SUFFIXES):
                    continue
                if not Path(dirpath, name).is_file():
                    continue
                found.append(str(Path(dirpath, name).relative_to(root)))
            if limit is not None and len(found) >= limit:
                break
        found.sort()
        if limit is not None:
            return found[:limit]
        return found


# --------------------------------------------------------------- completion
def _chat_messages(prompt: str, system: str) -> list[ChatCompletionMessageParam]:
    return [
        {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _anthropic_text(response: Any) -> str:
    """Concatenate every text block of an Anthropic response."""
    blocks = getattr(response, "content", None) or []
    parts = [
        block.text
        for block in blocks
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    if not parts:
        raise ValueError("Anthropic response contained no text content block")
    return "".join(parts)


def _openai_text(response: Any) -> str:
    """Extract the message text of an OpenAI-compatible (OpenAI/Ollama) response."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("OpenAI-compatible response contained no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not content:
        finish_reason = getattr(choices[0], "finish_reason", None)
        raise ValueError(
            f"OpenAI-compatible response returned empty content (finish_reason={finish_reason!r})"
        )
    return content


# ------------------------------------------------------------------- json
def extract_json(text: str) -> str:
    """Pull a JSON object/array out of an arbitrary LLM response.

    Handles bare JSON, ```json fenced blocks, prose preambles/postambles and
    trailing commas. Braces inside string values are respected. Raises
    ``ValueError`` (including the offending text) when no JSON can be found.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"extract_json: empty response, no JSON found in: {text!r}")

    body = _strip_code_fences(text.strip())
    spans = list(_iter_json_spans(body))
    if not spans:
        raise ValueError(f"extract_json: no JSON object or array found in: {text!r}")

    best: str | None = None
    fallback = max(spans, key=len)
    for span in spans:
        candidate = span
        try:
            json.loads(candidate)
        except ValueError:
            repaired = _strip_trailing_commas(candidate)
            if repaired == candidate:
                continue
            try:
                json.loads(repaired)
            except ValueError:
                continue
            candidate = repaired
        if best is None or len(candidate) > len(best):
            best = candidate
    return best if best is not None else fallback


def _strip_code_fences(text: str) -> str:
    """Return the contents of a single fenced block, else ``text`` unchanged."""
    blocks = _CODE_BLOCK_RE.findall(text)
    if len(blocks) == 1:
        return blocks[0].strip()
    if blocks:
        return text
    opening = _OPEN_FENCE_RE.match(text)
    # Unterminated fence (truncated response): drop the opening marker line.
    return text[opening.end() :] if opening else text


def _iter_json_spans(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}``/``[...]`` substring, ignoring string interiors."""
    index = 0
    length = len(text)
    while index < length:
        if text[index] not in _BRACKET_PAIRS:
            index += 1
            continue
        end = _scan_balanced(text, index)
        if end is None:
            index += 1
            continue
        yield text[index:end]
        index = end


def _scan_balanced(text: str, start: int) -> int | None:
    """Index just past the bracket matching ``text[start]``, or None if unbalanced."""
    stack = [_BRACKET_PAIRS[text[start]]]
    index = start + 1
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in _BRACKET_PAIRS:
            stack.append(_BRACKET_PAIRS[char])
        elif char in ("}", "]"):
            if stack.pop() != char:
                return None
            if not stack:
                return index + 1
        index += 1
    return None


def _strip_trailing_commas(text: str) -> str:
    """Remove commas that directly precede a closing brace/bracket, outside strings."""
    out: list[str] = []
    pending: int | None = None
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char.isspace():
            out.append(char)
            continue
        if char == '"':
            in_string = True
            pending = None
        elif pending is not None and char in "}]":
            out[pending] = ""
            pending = None
        else:
            pending = None
        out.append(char)
        if char == ",":
            pending = len(out) - 1
    return "".join(out)
