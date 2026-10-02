from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APIError as AnthropicAPIError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import AsyncAnthropic
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIError as OpenAIAPIError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings
from furrow.core.json_utils import extract_json

_MAX_ATTEMPTS = 3
_RATE_LIMIT_STATUS = 429


def _default_settings() -> Settings:
    """Return the module-level settings singleton, looked up at call time.

    Resolving this lazily (instead of as a default argument) means environment
    changes made after import are honoured.
    """
    return settings


def _is_transient(exc: BaseException) -> bool:
    """True for rate limits, timeouts, connection failures and 5xx API errors.

    Deliberately false for programming/config errors such as a missing API key,
    which are raised as ``ValueError`` and will never succeed on a retry.
    """
    if isinstance(
        exc,
        (
            AnthropicAPIConnectionError,
            OpenAIAPIConnectionError,
            AnthropicAPITimeoutError,
            OpenAIAPITimeoutError,
        ),
    ):
        return True
    if isinstance(exc, (AnthropicAPIError, OpenAIAPIError)):
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status == _RATE_LIMIT_STATUS or status >= 500
    return False


retry_transient = retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    reraise=True,
)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        # ``or`` keeps the keyword call shape used by the agents working while
        # falling back to the module singleton when None is passed.
        self.settings = settings or _default_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

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
        """Ollama exposes an OpenAI-compatible surface under ``/v1``."""
        if self._ollama is None:
            base_url = (self.settings.ollama_base_url or "").rstrip("/")
            if not base_url:
                raise ValueError("ollama_base_url is not set")
            # The key is a placeholder: Ollama ignores it but the SDK requires one.
            self._ollama = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        if self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def complete_json(self, prompt: str, system: str = "", model: str | None = None) -> Any:
        """Complete ``prompt`` and parse the first JSON value out of the response."""
        return extract_json(await self.complete(prompt, system=system, model=model))

    @retry_transient
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Content is a list of blocks (text, thinking, tool_use, ...); only the
        # text blocks carry output, and the list may be empty.
        return "".join(
            text
            for text in (getattr(block, "text", None) for block in response.content or [])
            if isinstance(text, str)
        )

    @retry_transient
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @retry_transient
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
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
