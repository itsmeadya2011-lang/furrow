from __future__ import annotations

import os
from pathlib import Path

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, get_settings


class LLMError(Exception):
    """Raised when LLM API calls fail after retries."""
    pass


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
            )
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(
                api_key=api_key,
                timeout=self.settings.llm_timeout,
            )
        return self._openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        try:
            if self.settings.provider == Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                return await self._complete_openai(prompt, system, model)
            else:
                raise LLMError(f"Unsupported provider: {self.settings.provider}")
        except (anthropic.APIError, openai.APIError) as e:
            raise LLMError(f"LLM API error: {e}") from e
        except (anthropic.APIConnectionError, openai.APIConnectionError) as e:
            raise LLMError(f"LLM connection error: {e}") from e
        except Exception as e:
            raise LLMError(f"Unexpected LLM error: {e}") from e

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, openai.RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            raise
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}") from e

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content or ""
        except openai.RateLimitError:
            raise
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e

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
