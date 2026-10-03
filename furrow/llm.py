"""LLM client layer plus the workspace file helpers the agents rely on.

Providers: Anthropic (native SDK) and OpenAI (native SDK), with Ollama served
through its OpenAI-compatible endpoint so only the base URL differs. Transient
failures (rate limits, timeouts, connection errors, 5xx) are retried a bounded
number of times; client errors such as a bad API key are surfaced immediately
because retrying them only wastes wall clock.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import aiofiles  # type: ignore[import-untyped]
import anthropic
import openai
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from furrow.config import MAX_TOKENS_DEFAULT, Provider, Settings, settings

# ``MAX_TOKENS_DEFAULT`` is imported from :mod:`furrow.config`, which owns it as
# the single source of truth for ``Settings.max_tokens``; re-exported through
# ``__all__`` so ``from furrow.llm import MAX_TOKENS_DEFAULT`` keeps working.
# The live value is always ``self.settings.max_tokens`` -- see ``LLMClient`` -- so
# ``FURROW_MAX_TOKENS`` wins over the constant.
__all__ = [
    "DEFAULT_IGNORE",
    "MAX_TOKENS_DEFAULT",
    "LLMClient",
    "anthropic_text",
    "is_retryable_error",
]

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."

RETRY_ATTEMPTS = 3

RETRY_MAX_WAIT = 8.0

OLLAMA_PLACEHOLDER_API_KEY = "ollama"

DEFAULT_IGNORE: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "*.pyc",
        ".venv",
        "venv",
        ".tox",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".furrow",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
    }
)


def anthropic_text(content: Iterable[object]) -> str:
    """Concatenate the text of every ``TextBlock`` in an Anthropic content list.

    ``response.content`` is heterogeneous: thinking, redacted-thinking,
    tool-use and server-tool blocks carry no ``.text`` attribute, and the list
    may be empty. Returns ``""`` when the model produced no text at all.
    """
    return "".join(block.text for block in content if isinstance(block, TextBlock))


def is_retryable_error(exc: BaseException) -> bool:
    """True for transient provider failures worth retrying.

    Rate limits, connection/timeout failures (both SDKs' ``APIConnectionError``
    covers ``APITimeoutError``) and server-side (5xx) errors are transient.
    Everything else -- 4xx, authentication, bad request -- is not: the same call
    will fail identically on the next attempt.
    """
    if isinstance(exc, (anthropic.RateLimitError, openai.RateLimitError)):
        return True
    if isinstance(exc, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return True
    if isinstance(exc, (anthropic.APIStatusError, openai.APIStatusError)):
        return exc.status_code is not None and exc.status_code >= 500
    return False


_TRANSIENT_RETRY = retry(
    retry=retry_if_exception(is_retryable_error),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_random_exponential(multiplier=0.5, max=RETRY_MAX_WAIT),
    reraise=True,
)


class LLMClient:
    """Provider-agnostic async completion client.

    Retries live here, on ``complete``, so every logical call -- whatever the
    provider -- shares a single bounded attempt budget.
    """

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def max_tokens(self) -> int:
        """Per-call token cap, read live from ``Settings.max_tokens``.

        Deliberately not :data:`MAX_TOKENS_DEFAULT`: the constant is only the
        default, so ``FURROW_MAX_TOKENS`` overrides it.
        """
        return self.settings.max_tokens

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is not set but the selected provider is "
                    f"{self._provider_name()!r}. Export ANTHROPIC_API_KEY, set "
                    "FURROW_ANTHROPIC_API_KEY, or pick another provider with "
                    f"FURROW_PROVIDER (current: {self._provider_name()})."
                )
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is not set but the selected provider is "
                    f"{self._provider_name()!r}. Export OPENAI_API_KEY, set "
                    "FURROW_OPENAI_API_KEY, or pick another provider with "
                    f"FURROW_PROVIDER (current: {self._provider_name()})."
                )
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """OpenAI-compatible client pointed at the local Ollama server.

        The OpenAI SDK rejects an empty API key, and Ollama ignores the value,
        so a stable placeholder is sent instead.
        """
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key=OLLAMA_PLACEHOLDER_API_KEY,
            )
        return self._ollama

    def _provider_name(self) -> str:
        provider = self.settings.provider
        return getattr(provider, "value", str(provider))

    @_TRANSIENT_RETRY
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Return the model's text for ``prompt``, retrying transient failures.

        Returns ``""`` when the provider replies with no text; callers parse the
        result as JSON and report an unhelpful error, so this stays a string
        rather than an exception.
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
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return anthropic_text(response.content)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        return await self._openai_chat(self.openai, prompt, system, model)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        return await self._openai_chat(self.ollama, prompt, system, model)

    async def _openai_chat(
        self,
        client: AsyncOpenAI,
        prompt: str,
        system: str,
        model: str,
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        target = Path(path)
        try:
            async with aiofiles.open(target, "r") as f:
                return await f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"read_file: no such file: {target.resolve()}") from exc

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write ``content`` to ``path`` atomically via a same-directory temp file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        os.close(handle)
        try:
            async with aiofiles.open(tmp_name, "w") as f:
                await f.write(content)
            if target.exists():
                shutil.copymode(target, tmp_name)
            else:
                os.chmod(tmp_name, 0o644)
            os.replace(tmp_name, target)
        except BaseException:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

    def list_files(
        self,
        directory: str | Path,
        ignore: Iterable[str] | None = None,
    ) -> list[str]:
        """List repository files as sorted POSIX-relative paths, skipping noise.

        ``DEFAULT_IGNORE`` (version-control metadata, tool caches, build output,
        virtualenvs) is always applied; ``ignore`` patterns extend it. Patterns
        match either a path's basename or its POSIX-relative path. A directory
        that does not exist yields ``[]``.
        """
        root = Path(directory)
        if not root.exists():
            return []
        patterns: tuple[str, ...] = tuple(DEFAULT_IGNORE)
        if ignore is not None:
            patterns = (*patterns, *ignore)

        def ignored(name: str, relative: str) -> bool:
            return any(
                fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(relative, pattern)
                for pattern in patterns
            )

        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not ignored(name, (current / name).relative_to(root).as_posix())
            )
            for name in sorted(filenames):
                relative = (current / name).relative_to(root).as_posix()
                if not ignored(name, relative):
                    found.append(relative)
        return sorted(found)