from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
import tenacity
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

log = structlog.get_logger()


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None

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
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        log.info("llm.complete", provider=self.settings.provider.value, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        log.debug("llm.anthropic.request", model=model, prompt_len=len(prompt))
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        log.debug("llm.anthropic.response", model=model)
        return response.content[0].text

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        log.debug("llm.openai.request", model=model, prompt_len=len(prompt))
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        log.debug("llm.openai.response", model=model)
        return response.choices[0].message.content or ""

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        log.debug("llm.ollama.request", model=model, prompt_len=len(prompt))
        url = f"{self.settings.ollama_base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        response = await self.http_client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        log.debug("llm.ollama.response", model=model)
        return data["choices"][0]["message"]["content"] or ""

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