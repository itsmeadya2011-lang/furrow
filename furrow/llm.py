from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiofiles
import httpx
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    AsyncAnthropic,
)
from anthropic import (
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
)
from openai import (
    AsyncOpenAI,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    AnthropicAPIConnectionError,
    AnthropicRateLimitError,
    OpenAIAPIConnectionError,
    OpenAIRateLimitError,
    httpx.TransportError,
    httpx.TimeoutException,
)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
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

    async def _with_retry(self, request: Callable[[], Awaitable[str]]) -> str:
        """Run an async request with exponential-backoff retries for transient errors.

        Retry count comes from the instance's settings, so custom Settings
        objects are respected.
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max(1, self.settings.max_retries)),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TRANSIENT_ERRORS),
            reraise=True,
        ):
            with attempt:
                return await request()

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
        async def request() -> str:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            )
            if not text.strip():
                raise ValueError("Empty response from Anthropic model")
            return text

        return await self._with_retry(request)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        async def request() -> str:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("Empty response from OpenAI model")
            return content

        return await self._with_retry(request)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        async def request() -> str:
            url = f"{self.settings.ollama_base_url.rstrip('/')}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "system": system or "You are a helpful coding assistant.",
                "stream": False,
                "options": {"num_predict": 4096},
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
            if response.status_code >= 400:
                raise ValueError(f"Ollama request failed: {response.text}")
            data = response.json()
            text = data.get("response", "")
            if not text.strip():
                raise ValueError("Empty response from Ollama model")
            return text

        return await self._with_retry(request)

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