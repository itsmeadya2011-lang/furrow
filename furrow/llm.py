from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
import structlog
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import AsyncAnthropic
from httpx import RequestError as HttpxRequestError
from openai import APIStatusError as OpenaiAPIStatusError
from openai import AsyncOpenAI
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


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
                raise ValueError(
                    f"ANTHROPIC_API_KEY is not set (provider={self.settings.provider})"
                )
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    f"OPENAI_API_KEY is not set (provider={self.settings.provider})"
                )
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @retry(
        retry=retry_if_exception_type(
            (AnthropicAPIStatusError, OpenaiAPIStatusError, HttpxRequestError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(4),
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.info(
            "llm.complete.requested",
            provider=self.settings.provider,
            model=model,
            system_preview=(system or "")[:100],
        )
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(
        retry=retry_if_exception_type(
            (AnthropicAPIStatusError, OpenaiAPIStatusError, HttpxRequestError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(4),
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        logger.info("llm.complete.anthropic", model=model)
        try:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            logger.info(
                "llm.complete.anthropic.success",
                model=model,
                usage=getattr(response, "usage", None),
            )
            return response.content[0].text
        except (AnthropicAPIStatusError, HttpxRequestError) as e:
            logger.error("llm.complete.anthropic.error", model=model, error=str(e))
            raise

    @retry(
        retry=retry_if_exception_type(
            (AnthropicAPIStatusError, OpenaiAPIStatusError, HttpxRequestError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(4),
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        logger.info("llm.complete.openai", model=model)
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            logger.info(
                "llm.complete.openai.success",
                model=model,
                usage=getattr(response, "usage", None),
            )
            return response.choices[0].message.content or ""
        except (OpenaiAPIStatusError, HttpxRequestError) as e:
            logger.error("llm.complete.openai.error", model=model, error=str(e))
            raise

    @retry(
        retry=retry_if_exception_type(
            (AnthropicAPIStatusError, OpenaiAPIStatusError, HttpxRequestError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(4),
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        import httpx

        logger.info("llm.complete.ollama", model=model)
        base_url = str(self.settings.ollama_base_url).rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=httpx.Timeout(120))
                response.raise_for_status()
            data = response.json()
            logger.info("llm.complete.ollama.success", model=model)
            return data.get("message", {}).get("content", "")
        except HttpxRequestError as e:
            logger.error("llm.complete.ollama.error", model=model, error=str(e))
            raise

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    async def delete_file(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists():
            p.unlink()

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
