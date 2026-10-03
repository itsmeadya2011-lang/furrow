from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import (
    APIConnectionError as AnthropicConnectionError,
)
from anthropic import (
    APITimeoutError as AnthropicTimeoutError,
)
from anthropic import (
    AsyncAnthropic,
)
from anthropic import (
    InternalServerError as AnthropicServerError,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

DEFAULT_MAX_TOKENS = 4096
DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."
OLLAMA_API_KEY = "ollama"
PREVIEW_CHARS = 200

# Transient failures worth retrying. 4xx validation/auth errors are excluded on purpose:
# retrying a bad API key only burns wall-clock time.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    AnthropicConnectionError,
    AnthropicTimeoutError,
    AnthropicRateLimitError,
    AnthropicServerError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

RETRY = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=5),
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
)


def extract_json(text: str) -> Any:
    """Parse JSON out of an LLM reply, tolerating a surrounding ```json fence."""
    candidate = text.strip()
    if candidate.startswith("```"):
        # Drop the opening fence (with its optional language tag) and the closing fence.
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate[3:]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
        candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        preview = candidate[:PREVIEW_CHARS]
        raise ValueError(
            f"LLM returned invalid JSON ({exc}); response preview: {preview!r}"
        ) from exc


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = default_settings if settings is None else settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "Anthropic API key is missing: set FURROW_ANTHROPIC_API_KEY "
                    "or ANTHROPIC_API_KEY"
                )
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key is missing: set FURROW_OPENAI_API_KEY or OPENAI_API_KEY"
                )
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """Ollama serves an OpenAI-compatible API and ignores the key."""
        if self._ollama is None:
            base_url = f"{self.settings.ollama_base_url.rstrip('/')}/v1"
            self._ollama = AsyncOpenAI(base_url=base_url, api_key=OLLAMA_API_KEY)
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

    @RETRY
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=system or DEFAULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_chat(self.openai, prompt, system, model)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_chat(self.ollama, prompt, system, model)

    @RETRY
    async def _complete_chat(
        self, client: AsyncOpenAI, prompt: str, system: str, model: str
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(Path(path), "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    async def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
