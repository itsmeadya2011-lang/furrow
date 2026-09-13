from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiofiles
import httpx
from anthropic import AsyncAnthropic, AuthenticationError as AnthropicAuthError, BadRequestError as AnthropicBadRequestError
from openai import AsyncOpenAI, AuthenticationError as OpenAIAuthError, BadRequestError as OpenAIBadRequestError

from furrow.config import Provider, Settings


class NonRetryableError(Exception):
    """Error that should not be retried (e.g., auth, validation)."""


def _is_retryable_error(exc: Exception) -> bool:
    # Programming errors and response-validation failures are not transient.
    if isinstance(exc, (ValueError, TypeError)):
        return False
    # Provider-level auth/validation errors are not transient.
    if isinstance(
        exc,
        (
            AnthropicAuthError,
            OpenAIAuthError,
            AnthropicBadRequestError,
            OpenAIBadRequestError,
            NonRetryableError,
        ),
    ):
        return False
    # HTTP 4xx client errors (e.g. OLLAMA) are not transient; 5xx and
    # connection/timeout errors remain retryable.
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return True


async def _retry_with_backoff(
    func: Callable[..., Awaitable[Any]],
    *args: Any,
    max_retries: int,
    base_delay: float,
    timeout: float,
    **kwargs: Any,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable_error(exc):
                break
            delay = base_delay * (2**attempt)
            await asyncio.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry loop exited without result or exception")


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        from furrow.config import settings as global_settings

        self.settings = settings or global_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.get_anthropic_key() or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.get_openai_key() or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def httpx(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(timeout=self.settings.request_timeout)
        return self._httpx

    async def close(self) -> None:
        # Best-effort: ensure every initialized client is closed even if one
        # raises, so a single failure does not prevent the others from closing.
        async def _safe_close(coro) -> None:
            try:
                await coro
            except Exception:
                pass

        if self._anthropic is not None:
            await _safe_close(self._anthropic.close())
            self._anthropic = None
        if self._openai is not None:
            await _safe_close(self._openai.close())
            self._openai = None
        if self._httpx is not None:
            await _safe_close(self._httpx.aclose())
            self._httpx = None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        async def _call() -> str:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            content_blocks = response.content or []
            if not content_blocks:
                return ""
            text_parts = [block.text for block in content_blocks if getattr(block, "type", "") == "text" and block.text]
            return "\n".join(text_parts) if text_parts else ""

        return await _retry_with_backoff(
            _call,
            max_retries=self.settings.max_retries,
            base_delay=self.settings.retry_base_delay,
            timeout=self.settings.request_timeout,
        )

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        async def _call() -> str:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            choices = response.choices or []
            if not choices:
                return ""
            message = choices[0].message if choices[0] else None
            if message is None:
                return ""
            return message.content or ""

        return await _retry_with_backoff(
            _call,
            max_retries=self.settings.max_retries,
            base_delay=self.settings.retry_base_delay,
            timeout=self.settings.request_timeout,
        )

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        async def _call() -> str:
            response = await self.httpx.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"OLLAMA response is not a JSON object: {type(data).__name__}")
            message = data.get("message") or {}
            if not isinstance(message, dict):
                raise ValueError(f"OLLAMA response 'message' is not a JSON object: {type(message).__name__}")
            content = message.get("content")
            if content is None:
                return ""
            return str(content)

        return await _retry_with_backoff(
            _call,
            max_retries=self.settings.max_retries,
            base_delay=self.settings.retry_base_delay,
            timeout=self.settings.request_timeout,
        )

    def _resolve_path(self, path: str | Path) -> Path:
        target = Path(path)
        workspace = self.settings.workspace.resolve()
        if target.is_absolute():
            try:
                resolved = target.resolve()
                resolved.relative_to(workspace)
            except ValueError:
                raise ValueError(f"Absolute path escapes workspace: {path}") from None
        else:
            try:
                resolved = (workspace / target).resolve()
                resolved.relative_to(workspace)
            except ValueError:
                raise ValueError(f"Path escapes workspace: {path}") from None
        return resolved

    async def read_file(self, path: str | Path) -> str:
        resolved = self._resolve_path(path)
        async with aiofiles.open(resolved, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(resolved, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        resolved = self._resolve_path(directory)
        if not resolved.exists():
            return []
        files = []
        for f in resolved.rglob("*"):
            if f.is_file():
                try:
                    rel = f.relative_to(resolved)
                    files.append(str(rel))
                except ValueError:
                    continue
        return sorted(files)
