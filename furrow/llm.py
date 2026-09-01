from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._retry_config = AsyncRetrying(
            stop=stop_after_attempt(settings.max_retries),
            wait=wait_exponential(multiplier=settings.retry_base_delay),
            retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError, ValueError)),
            reraise=True,
        )

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
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(base_url=self.settings.ollama_base_url)
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("llm.complete", provider=self.settings.provider.value, model=model)

        async def _do_complete() -> str:
            if self.settings.provider == Provider.ANTHROPIC:
                return await self._complete_anthropic(prompt, system, model)
            elif self.settings.provider == Provider.OPENAI:
                return await self._complete_openai(prompt, system, model)
            elif self.settings.provider == Provider.OLLAMA:
                return await self._complete_ollama(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")

        try:
            result = await asyncio.wait_for(
                self._retry_config(_do_complete),
                timeout=float(self.settings.llm_timeout),
            )
        except asyncio.TimeoutError:
            logger.error("llm.timeout", provider=self.settings.provider.value, model=model)
            raise TimeoutError(
                f"LLM call timed out after {self.settings.llm_timeout}s "
                f"({self.settings.provider.value}/{model})"
            )
        logger.debug("llm.complete.done", provider=self.settings.provider.value, model=model)
        return result

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        payload = {
            "model": model or self.settings.ollama_model,
            "prompt": prompt,
            "system": system or "You are a helpful coding assistant.",
            "stream": False,
            "options": {"temperature": 0.7},
        }
        response = await self.http_client.post(
            "/api/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.settings.workspace) / p
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
