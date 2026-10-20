from __future__ import annotations

import os
from pathlib import Path

import aiofiles
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, RetryCallState, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

MAX_ATTEMPTS = 3

_RETRYABLE_TOKENS = (
    "Timeout",
    "Connection",
    "RateLimit",
    "InternalServer",
    "ServiceUnavailable",
    "Overloaded",
    "APIStatus",
)

_NON_RETRYABLE_TYPES = (ValueError, TypeError, KeyError)


def _is_retryable(retry_state: RetryCallState) -> bool:
    """Return True if the failure in ``retry_state`` is worth retrying.

    SDK exception classes differ between versions, so we match on the
    exception class name instead of importing SDK error types.
    ValueError/TypeError/KeyError (e.g. a missing API key) are never
    retried so configuration failures fail fast.
    """
    outcome = retry_state.outcome
    if outcome is None:
        return False
    exc = outcome.exception()
    if exc is None or isinstance(exc, _NON_RETRYABLE_TYPES):
        return False
    name = type(exc).__name__
    return any(token in name for token in _RETRYABLE_TOKENS)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

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

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        requested = model
        if self.settings.provider == Provider.ANTHROPIC:
            resolved = requested or self.settings.model
            call = lambda: self._complete_anthropic(prompt, system, resolved)
        elif self.settings.provider == Provider.OPENAI:
            resolved = requested or self.settings.model
            call = lambda: self._complete_openai(prompt, system, resolved)
        elif self.settings.provider == Provider.OLLAMA:
            resolved = requested or getattr(self.settings, "ollama_model", "") or self.settings.model
            call = lambda: self._complete_ollama(prompt, system, resolved)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=_is_retryable,
            reraise=True,
        ):
            with attempt:
                return await call()
        raise AssertionError("retry loop exited without returning")  # pragma: no cover

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.content
        if isinstance(content, str):
            # Tolerate mocked responses whose content is a plain string.
            return content
        texts: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        return "\n".join(texts)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        return content if isinstance(content, str) else ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=body)
        payload = response.json()
        data = payload if isinstance(payload, dict) else {}
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        return content if isinstance(content, str) else ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
