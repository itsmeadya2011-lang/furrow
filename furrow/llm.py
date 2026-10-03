from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

# Module-level alias so ``__init__`` can reach the singleton: the ``settings``
# parameter shadows the imported name inside the method body, and capturing the
# singleton as a default argument would freeze it at import time.
_DEFAULT_SETTINGS: Final[Settings] = settings

DEFAULT_SYSTEM_PROMPT: Final[str] = "You are a helpful coding assistant."

IGNORED_DIRS: Final[frozenset[str]] = frozenset(
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
        ".tox",
    }
)

_MAX_SNIPPET_CHARS: Final[int] = 300


class LLMResponseError(ValueError):
    """Raised when a model response cannot be parsed into the expected JSON shape."""


def _snippet(text: str, limit: int = _MAX_SNIPPET_CHARS) -> str:
    """Return a short, single-line excerpt of ``text`` suitable for error messages."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "..."


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence (``` or ```json) from ``text``."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    newline = stripped.find("\n")
    if newline == -1:
        return text
    body = stripped[newline + 1 :]
    closing = body.rfind("```")
    if closing == -1:
        return body
    return body[:closing]


def _scan_balanced_object(text: str, start: int) -> int | None:
    """Return the index of the ``}`` closing the ``{`` at ``start``, or ``None``.

    Braces and quotes inside JSON string literals are ignored, and backslash
    escapes are honoured so ``{"a": "}{"}`` scans correctly.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` region of ``text`` as a JSON string.

    Markdown code fences are stripped first, so fenced or prose-wrapped JSON is
    recovered. Braces and quotes inside JSON strings are respected, and nested
    objects and arrays work. Regions that are not valid JSON are skipped so stray
    braces in prose do not shadow the real payload; if nothing decodes, the first
    balanced region is returned so the caller can report the decode failure. Raises
    ``LLMResponseError`` when no balanced object exists at all.
    """
    candidate = _strip_code_fences(text)
    start = candidate.find("{")
    first_region: str | None = None
    while start != -1:
        end = _scan_balanced_object(candidate, start)
        if end is not None:
            region = candidate[start : end + 1]
            if first_region is None:
                first_region = region
            try:
                return json.dumps(json.loads(region))
            except json.JSONDecodeError:
                pass
        start = candidate.find("{", start + 1)
    if first_region is not None:
        return first_region
    raise LLMResponseError(f"No JSON object found in response: {_snippet(text)!r}")


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object embedded in ``text`` and return it as a dict.

    Raises ``LLMResponseError`` (a ``ValueError``) with a truncated excerpt of the
    original response when extraction, decoding, or the dict check fails.
    """
    raw = extract_json_object(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"Could not decode JSON from response ({exc}); payload={_snippet(raw)!r}"
        ) from exc
    if not isinstance(data, dict):
        raise LLMResponseError(
            f"Expected a JSON object but got {type(data).__name__}; "
            f"payload={_snippet(raw)!r}"
        )
    return data


class LLMClient:
    """Thin, dependency-light wrapper around the supported chat completion providers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings if settings is not None else _DEFAULT_SETTINGS
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazily built Anthropic client configured with the configured timeout."""
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(
                api_key=api_key, timeout=self.settings.request_timeout
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazily built OpenAI client configured with the configured timeout."""
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key, timeout=self.settings.request_timeout)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """Lazily built OpenAI-compatible client pointed at the local Ollama server."""
        if self._ollama is None:
            api_key = self.settings.openai_api_key or os.getenv("OLLAMA_API_KEY") or "ollama"
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key=api_key,
                timeout=self.settings.request_timeout,
            )
        return self._ollama

    def _with_retry(
        self, func: Callable[..., Awaitable[str]]
    ) -> Callable[..., Awaitable[str]]:
        """Wrap a provider coroutine with the retry policy from the current settings.

        Policy is resolved per call so mutating ``self.settings.llm_max_retries``
        takes effect immediately. Parse errors (``LLMResponseError`` and any other
        ``ValueError``) are never retried.
        """
        return retry(
            stop=stop_after_attempt(max(1, self.settings.llm_max_retries)),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_not_exception_type((ValueError, LLMResponseError)),
            reraise=True,
        )(func)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Run a single chat completion and return the assistant text."""
        model = model or self.settings.model
        provider = self.settings.provider
        handlers: dict[Provider, Callable[[str, str, str], Awaitable[str]]] = {
            Provider.ANTHROPIC: self._complete_anthropic,
            Provider.OPENAI: self._complete_openai,
            Provider.OLLAMA: self._complete_ollama,
        }
        try:
            handler = handlers[Provider(provider)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unsupported provider: {provider!r}") from exc
        return await self._with_retry(handler)(prompt, system, model)

    async def complete_json(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> dict[str, Any]:
        """Run a completion and parse the response as a JSON object."""
        return parse_json_object(await self.complete(prompt, system=system, model=model))

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in list(getattr(response, "content", None) or []):
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                return text
        return ""

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        return await self._chat_completion(self.openai, prompt, system, model)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        return await self._chat_completion(self.ollama, prompt, system, model)

    async def _chat_completion(
        self, client: AsyncOpenAI, prompt: str, system: str, model: str
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        choices = list(getattr(response, "choices", None) or [])
        if not choices:
            return ""
        return getattr(choices[0].message, "content", None) or ""

    async def read_file(self, path: str | Path) -> str:
        """Read and return the UTF-8 contents of ``path``."""
        async with aiofiles.open(path, encoding="utf-8") as handle:
            return await handle.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write ``content`` to ``path`` as UTF-8, creating parent directories."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as handle:
            await handle.write(content)

    def list_files(self, directory: str | Path | None = None) -> list[str]:
        """Return sorted repo-relative file paths below ``directory`` (default: workspace).

        Hidden entries and the directories in ``IGNORED_DIRS`` are skipped so the
        result stays usable on a real checkout.
        """
        root = Path(directory) if directory is not None else Path(self.settings.workspace)
        if not root.is_dir():
            return []
        files: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part.startswith(".") or part in IGNORED_DIRS for part in relative.parts):
                continue
            files.append(relative.as_posix())
        return sorted(files)