"""Async LLM client for furrow.

``LLMClient`` lazily builds one SDK client per provider, applies an explicit
request timeout, and can be closed explicitly (``aclose()``) or used as an async
context manager so the underlying httpx connections are released.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Final

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.config import settings as global_settings

DEFAULT_SYSTEM_PROMPT: Final[str] = "You are a helpful coding assistant."

#: Directories skipped by :meth:`LLMClient.list_files`.
IGNORED_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
    }
)

#: File suffixes skipped by :meth:`LLMClient.list_files`.
IGNORED_SUFFIXES: Final[tuple[str, ...]] = (".pyc", ".pyo")


__all__ = ["LLMClient", "LLMError", "LLMResponseError"]


class LLMError(RuntimeError):
    """A provider call failed or returned something unusable."""


class LLMResponseError(LLMError, ValueError):
    """A provider response was empty, unparsable, truncated, or not the expected shape.

    This is the single canonical definition of the error; ``furrow.agents.planner``
    imports it rather than defining a second, unrelated class.

    ``ValueError`` is a deliberate second base: callers that already guard parsing
    with ``except ValueError`` (e.g. around ``extract_json_object`` plus
    ``TestResult(**data)`` in ``furrow.agents.tester``) keep catching a malformed
    model reply.  ``LLMError`` keeps it a ``RuntimeError`` for transport-style
    handling.  The MRO is consistent: LLMResponseError -> LLMError -> RuntimeError
    -> ValueError -> Exception.
    """


def _preview(text: str, limit: int = 200) -> str:
    """Collapse *text* and clip it to *limit* characters for error messages."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def _extract_anthropic_text(content: object) -> str:
    """Return the text of the first usable text block of an Anthropic response.

    Non-text blocks (``thinking``, ``tool_use``, ``redacted_thinking``, ...) are
    skipped instead of blowing up, and a response that carries no text at all
    raises :class:`LLMResponseError` instead of ``IndexError``/``AttributeError``.
    """
    blocks = content if isinstance(content, (list, tuple)) else []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            return text
    seen = ", ".join(sorted({type(block).__name__ for block in blocks})) or "none"
    raise LLMResponseError(
        f"Anthropic response contained no text block (block types: {seen}). "
        "The model may have returned tool calls or thinking blocks only."
    )


def _scan_files(root: Path) -> list[str]:
    """Blocking part of :meth:`LLMClient.list_files`; runs in a worker thread."""
    resolved_root = root.resolve()
    found: list[str] = []
    for entry in root.rglob("*"):
        if entry.suffix in IGNORED_SUFFIXES:
            continue
        try:
            relative = entry.relative_to(root)
        except ValueError:  # pragma: no cover - rglob only yields paths below root
            continue
        if any(part in IGNORED_DIR_NAMES for part in relative.parts[:-1]):
            continue
        if entry.is_symlink():
            # Skip links pointing outside the scanned tree (relative_to raises ValueError).
            try:
                entry.resolve().relative_to(resolved_root)
            except (OSError, ValueError):
                continue
        if not entry.is_file():
            continue
        found.append(str(relative))
    return sorted(found)


class LLMClient:
    """Thin async wrapper around the supported LLM providers."""

    def __init__(self, settings: Settings | None = None) -> None:
        # ``furrow/agents/*`` forwards ``settings=settings`` where the agent's own
        # parameter defaults to None, so an explicit None must fall back to the
        # module singleton instead of being stored verbatim.
        self.settings: Settings = settings if settings is not None else global_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        # Guards lazy construction: the orchestrator fans out N worker coroutines
        # that share one client.  asyncio.Lock binds to the running loop on first
        # use, so building it here (possibly without a loop) is safe on 3.10+.
        self._client_lock = asyncio.Lock()

    # ------------------------------------------------------------- SDK clients

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazily built ``AsyncAnthropic`` client (thread-safe, no ``await``)."""
        client = self._anthropic
        if client is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            client = AsyncAnthropic(api_key=api_key, timeout=self.settings.request_timeout)
            self._anthropic = client
        return client

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazily built ``AsyncOpenAI`` client (thread-safe, no ``await``)."""
        client = self._openai
        if client is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            client = AsyncOpenAI(api_key=api_key, timeout=self.settings.request_timeout)
            self._openai = client
        return client

    async def _anthropic_client(self) -> AsyncAnthropic:
        """Anthropic client, built under the lock if it does not exist yet."""
        async with self._client_lock:
            return self.anthropic

    async def _openai_client(self) -> AsyncOpenAI:
        """OpenAI client, built under the lock if it does not exist yet."""
        async with self._client_lock:
            return self.openai

    # --------------------------------------------------------------- lifecycle

    async def aclose(self) -> None:
        """Close whichever provider clients were created; safe to call twice."""
        async with self._client_lock:
            if self._anthropic is not None:
                await self._anthropic.close()
                self._anthropic = None
            if self._openai is not None:
                await self._openai.close()
                self._openai = None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # --------------------------------------------------------------- completion

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Run a single completion and return its text.

        Raises:
            LLMError: transport/API failure, or a response that is empty or
                truncated (truncation is surfaced explicitly instead of being
                returned as if it were a complete answer).
            ValueError: the configured provider has no usable credentials.
        """
        resolved_model = model or self.settings.model
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, resolved_model)
        if provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, resolved_model)
        if provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, resolved_model)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        client = await self._anthropic_client()
        response = await client.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "max_tokens":
            raise LLMResponseError(
                f"Anthropic response for model {model!r} was truncated "
                f"(stop_reason='max_tokens', max_tokens={self.settings.max_tokens}); "
                "raise Settings.max_tokens or shorten the prompt."
            )
        return _extract_anthropic_text(response.content)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        client = await self._openai_client()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        if not response.choices:
            raise LLMResponseError(f"OpenAI returned no choices for model {model!r}.")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMResponseError(
                f"OpenAI response for model {model!r} was truncated "
                f"(finish_reason='length', max_tokens={self.settings.max_tokens}); "
                "raise Settings.max_tokens or shorten the prompt."
            )
        content = choice.message.content
        if not content or not content.strip():
            details = [f"finish_reason={choice.finish_reason!r}"]
            refusal = getattr(choice.message, "refusal", None)
            if refusal:
                details.append(f"refusal={refusal!r}")
            tool_calls = getattr(choice.message, "tool_calls", None)
            if tool_calls:
                details.append(f"tool_calls={len(tool_calls)}")
            raise LLMResponseError(
                f"OpenAI returned no text content for model {model!r} ({', '.join(details)})."
            )
        return content

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        """Local-model convenience path against a plain Ollama server."""
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"num_predict": self.settings.max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout) as http:
                response = await http.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request to {url} failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(
                f"Ollama returned HTTP {response.status_code} for model {model!r} at {url}: "
                f"{_preview(response.text, 500)}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(
                f"Ollama returned a non-JSON body from {url}: {_preview(response.text, 500)!r}"
            ) from exc
        if not isinstance(data, dict):
            raise LLMError(
                f"Ollama returned {type(data).__name__} instead of a JSON object from {url}."
            )
        if data.get("done_reason") == "length":
            raise LLMResponseError(
                f"Ollama response for model {model!r} was truncated "
                f"(done_reason='length', max_tokens={self.settings.max_tokens}); "
                "raise Settings.max_tokens or shorten the prompt."
            )
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError(
                f"Ollama response from {url} has no 'message.content' for model {model!r}: "
                f"{_preview(response.text, 500)!r}"
            )
        return content

    # ------------------------------------------------------------------- files

    async def read_file(self, path: str | Path) -> str:
        """Read a text file as UTF-8 (never the locale-dependent default)."""
        async with aiofiles.open(path, encoding="utf-8") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Atomically overwrite a UTF-8 text file.

        The content goes to a temporary file in the same directory and is then
        ``os.replace``d into position, so a concurrent reader never observes a
        half-written file and a crash mid-write cannot truncate the original.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(content)
            os.replace(tmp_path, target)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def list_files(self, directory: str | Path) -> list[str]:
        """List files below *directory*, relative to it, without tooling noise.

        The scan runs in a worker thread so it cannot block the event loop.
        Noise directories (``.git``, ``__pycache__``, ``node_modules``, ...),
        compiled ``*.pyc`` files and symlinks escaping *directory* are skipped.
        """
        root = Path(directory)
        if not root.is_dir():
            return []
        return await asyncio.to_thread(_scan_files, root)