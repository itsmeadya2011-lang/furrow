from __future__ import annotations

import asyncio
import fnmatch
import os
import random
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow import config as config_module
from furrow.config import Provider, Settings
from furrow.util import JSONExtractionError, PathEscapeError, extract_json, resolve_within, truncate

__all__ = ["LLMClient", "LLMError"]

_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    asyncio.TimeoutError,
    TimeoutError,
)

_FATAL_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.BadRequestError,
    anthropic.NotFoundError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
    openai.NotFoundError,
)

_DEFAULT_SYSTEM = "You are a helpful coding assistant."
_BINARY_SNIFF_BYTES = 4096
_MAX_BACKOFF_SECONDS = 30.0


class LLMError(RuntimeError):
    """Raised when an LLM provider call cannot be completed."""


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings if settings is not None else config_module.settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key, timeout=self.settings.request_timeout)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key, timeout=self.settings.request_timeout)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """OpenAI-compatible client pointed at a local Ollama server."""
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
                timeout=self.settings.request_timeout,
            )
        return self._ollama

    def default_model(self) -> str:
        if self.settings.provider == Provider.OLLAMA:
            return self.settings.ollama_model
        return self.settings.model

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Run a single completion, retrying transient provider failures."""
        provider = self.settings.provider
        model = model or self.default_model()
        max_tokens = max_tokens or self.settings.max_tokens
        temperature = temperature if temperature is not None else self.settings.temperature
        attempts = max(1, self.settings.max_retries)

        last_error: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                if provider == Provider.ANTHROPIC:
                    return await self._complete_anthropic(prompt, system, model, max_tokens, temperature)
                if provider == Provider.OPENAI:
                    return await self._complete_openai(
                        self.openai, prompt, system, model, max_tokens, temperature
                    )
                if provider == Provider.OLLAMA:
                    return await self._complete_openai(
                        self.ollama, prompt, system, model, max_tokens, temperature
                    )
                raise LLMError(f"Unsupported provider: {provider!r}")
            except LLMError:
                raise
            except _FATAL_ERRORS as exc:
                raise LLMError(f"{provider} request failed: {exc}") from exc
            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                await asyncio.sleep(self._backoff(attempt))
            except Exception as exc:
                raise LLMError(f"{provider} request failed: {exc}") from exc

        raise LLMError(f"{provider} request failed after {attempts} attempts: {last_error}")

    def _backoff(self, attempt: int) -> float:
        base = max(0.0, self.settings.retry_base_delay)
        delay = min(base * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS)
        return delay + random.uniform(0.0, base)

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, max_tokens: int, temperature: float
    ) -> str:
        # ``temperature`` is accepted for signature symmetry but not sent: anthropic>=1.11
        # removed it from ``messages.create``.
        del temperature
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or _DEFAULT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        parts: list[str] = []
        for block in response.content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        if not parts:
            raise LLMError("anthropic returned no text content")
        return "".join(parts)

    async def _complete_openai(
        self,
        client: AsyncOpenAI,
        prompt: str,
        system: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or _DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError(f"{model} returned no choices")
        return getattr(choices[0].message, "content", None) or ""

    async def complete_json(self, prompt: str, system: str = "", model: str | None = None) -> Any:
        text = await self.complete(prompt, system=system, model=model)
        try:
            return extract_json(text)
        except JSONExtractionError as exc:
            snippet = truncate(text, limit=400, marker=" ... ")
            raise LLMError(f"could not parse JSON from model output: {exc}; output={snippet!r}") from exc

    async def read_file(self, path: str | Path, *, max_bytes: int | None = None) -> str:
        async with aiofiles.open(path) as f:
            content = await f.read()
        if max_bytes is not None and len(content) > max_bytes:
            content = truncate(content, limit=max_bytes)
        return content

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(
        self,
        directory: str | Path | None = None,
        *,
        max_files: int | None = None,
        include_dirs: bool = False,
    ) -> list[str]:
        root = Path(directory) if directory is not None else self.settings.workspace
        try:
            root = root.expanduser().resolve()
        except OSError:
            return []
        if not root.is_dir():
            return []

        limit = min(max_files or self.settings.max_file_list, self.settings.max_file_list)
        ignored_dirs = set(self.settings.ignore_dirs)
        ignore_globs = list(self.settings.ignore_globs)
        found: list[str] = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in ignored_dirs)
            base = Path(dirpath)
            if include_dirs:
                for name in dirnames:
                    found.append(str((base / name).relative_to(root)))
            for name in sorted(filenames):
                path = base / name
                relative = path.relative_to(root)
                if any(part in ignored_dirs for part in relative.parts):
                    continue
                if any(fnmatch.fnmatch(name, pattern) for pattern in ignore_globs):
                    continue
                try:
                    resolve_within(root, path)
                except (PathEscapeError, OSError):
                    continue
                if self._is_binary(path):
                    continue
                found.append(str(relative))

        found.sort()
        return found[:limit]

    @staticmethod
    def _is_binary(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                return b"\x00" in f.read(_BINARY_SNIFF_BYTES)
        except OSError:
            return True

    async def aclose(self) -> None:
        for attr in ("_anthropic", "_openai", "_ollama"):
            client = getattr(self, attr, None)
            if client is None:
                continue
            try:
                await client.close()
            except Exception:
                pass
            setattr(self, attr, None)
