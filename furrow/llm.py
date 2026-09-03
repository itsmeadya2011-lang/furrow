from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from anthropic.types import APIError as AnthropicAPIError
from openai import AsyncOpenAI
from openai.types import APIError as OpenAIAPIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from furrow.config import Provider, Settings, settings


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

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None, timeout: int = 120
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, timeout)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, timeout)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, timeout)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((AnthropicAPIError, Exception)),
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str, timeout: int) -> str:
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            return response.content[0].text
        except AnthropicAPIError as e:
            raise RuntimeError(f"Anthropic API error (model={model}): {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Anthropic (model={model}): {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OpenAIAPIError, Exception)),
    )
    async def _complete_openai(self, prompt: str, system: str, model: str, timeout: int) -> str:
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                timeout=timeout,
            )
            return response.choices[0].message.content or ""
        except OpenAIAPIError as e:
            raise RuntimeError(f"OpenAI API error (model={model}): {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling OpenAI (model={model}): {e}") from e

    async def _complete_ollama(self, prompt: str, system: str, model: str, timeout: int) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama API error (model={model}, url={url}): {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Ollama (model={model}): {e}") from e

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
