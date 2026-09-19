from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    exception_name = type(exc).__name__.lower()
    transient_markers = (
        "timeout",
        "connection",
        "rate_limit",
        "ratelimit",
        "server_error",
        "api_connection",
        "api_timeout",
    )
    return any(marker in exception_name for marker in transient_markers)


def _positive_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _positive_int(value: object, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _anthropic_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    try:
        iterator = iter(content)
    except TypeError:
        return ""
    for block in iterator:
        if isinstance(block, str):
            parts.append(block)
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _openai_content(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if not content:
        return ""

    parts: list[str] = []
    try:
        iterator = iter(content)
    except TypeError:
        return str(content)
    for block in iterator:
        if isinstance(block, str):
            parts.append(block)
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _openai_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None:
        return ""
    try:
        iterator = iter(choices)
    except TypeError:
        return ""
    for choice in iterator:
        content = _openai_content(choice)
        if content:
            return content
    return ""


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def timeout(self) -> float:
        return _positive_float(getattr(self.settings, "llm_timeout", 120.0), 120.0)

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key, timeout=self.timeout)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key, timeout=self.timeout)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        if self._ollama is None:
            base_url = str(
                getattr(self.settings, "ollama_base_url", "http://localhost:11434")
            ).rstrip("/")
            self._ollama = AsyncOpenAI(
                api_key="ollama",
                base_url=f"{base_url}/v1",
                timeout=self.timeout,
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        max_attempts = _positive_int(getattr(self.settings, "max_attempts", 2), 2)
        retry_delay = _positive_float(getattr(self.settings, "retry_delay", 1.0), 1.0)
        last_exception: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                if self.settings.provider == Provider.ANTHROPIC:
                    result = await asyncio.wait_for(
                        self._complete_anthropic(prompt, system, model),
                        timeout=self.timeout,
                    )
                elif self.settings.provider == Provider.OPENAI:
                    result = await asyncio.wait_for(
                        self._complete_openai(prompt, system, model),
                        timeout=self.timeout,
                    )
                elif self.settings.provider == Provider.OLLAMA:
                    result = await asyncio.wait_for(
                        self._complete_ollama(prompt, system, model),
                        timeout=self.timeout,
                    )
                else:
                    raise ValueError(f"Unsupported provider: {self.settings.provider}")

                if result is None:
                    return ""
                return result
            except asyncio.CancelledError:
                raise
            except ValueError:
                raise
            except Exception as exc:
                last_exception = exc
                if not _is_transient_error(exc) or attempt >= max_attempts:
                    raise
                await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))

        raise RuntimeError(f"LLM request failed after {max_attempts} attempts") from last_exception

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return _anthropic_text(response)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return _openai_text(response)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return _openai_text(response)

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as file:
            return await file.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w") as file:
            await file.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        path = Path(directory)
        if not path.exists():
            return []
        return [str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()]

    async def aclose(self) -> None:
        for client in (self._anthropic, self._openai, self._ollama):
            if client is not None:
                await client.close()
