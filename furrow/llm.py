from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings
from furrow.logging import get_logger

logger = get_logger("furrow.llm")


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama_client: httpx.AsyncClient | None = None

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
    def ollama(self) -> httpx.AsyncClient:
        if self._ollama_client is None:
            self._ollama_client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=120.0,
            )
        return self._ollama_client

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("llm_complete", provider=self.settings.provider.value, model=model, prompt_length=len(prompt))
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text
        logger.debug("llm_response", provider="anthropic", model=model, response_length=len(result))
        return result

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        result = response.choices[0].message.content or ""
        logger.debug("llm_response", provider="openai", model=model, response_length=len(result))
        return result

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.ollama.post(
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("message", {}).get("content", "")
        logger.debug("llm_response", provider="ollama", model=model, response_length=len(result))
        return result

    async def read_file(self, path: str | Path) -> str:
        logger.debug("read_file", path=str(path))
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        logger.debug("read_file_complete", path=str(path), length=len(content))
        return content

    async def write_file(self, path: str | Path, content: str) -> None:
        logger.debug("write_file", path=str(path), length=len(content))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        logger.info("write_file_complete", path=str(path))

    def list_files(self, directory: str | Path) -> list[str]:
        logger.debug("list_files", directory=str(directory))
        p = Path(directory)
        if not p.exists():
            return []
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
        logger.debug("list_files_complete", directory=str(directory), count=len(files))
        return files
