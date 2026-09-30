from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings
from furrow.config import settings as _default_settings

logger = structlog.get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else _default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

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
    def httpx_client(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(timeout=None)
        return self._httpx

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        timeout: float = 60.0,
    ) -> str:
        model = model or self.settings.model
        logger.info("llm_call", provider=self.settings.provider.value, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            response = await asyncio.wait_for(
                self._complete_anthropic(prompt, system, model),
                timeout=timeout,
            )
        elif self.settings.provider == Provider.OPENAI:
            response = await asyncio.wait_for(
                self._complete_openai(prompt, system, model),
                timeout=timeout,
            )
        elif self.settings.provider == Provider.OLLAMA:
            response = await asyncio.wait_for(
                self._complete_ollama(prompt, system, model),
                timeout=timeout,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")
        logger.info(
            "llm_response",
            provider=self.settings.provider.value,
            model=model,
            response_length=len(response),
        )
        return response

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # When no tools are configured, the first content block is always a TextBlock.
        content_block = response.content[0]
        if hasattr(content_block, "text"):
            return str(content_block.text)
        raise ValueError(f"Unexpected content block type: {type(content_block).__name__}")

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
        messages = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        async with self.httpx_client.stream(
            "POST",
            f"{self.settings.ollama_base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": True},
        ) as response:
            response.raise_for_status()
            content = ""
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("message", {}).get("content"):
                    content = chunk["message"]["content"]
            return content

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    async def edit_file(self, path: str | Path, old_string: str, new_string: str) -> bool:
        content = await self.read_file(path)
        if old_string not in content:
            return False
        content = content.replace(old_string, new_string, 1)
        await self.write_file(path, content)
        return True

    async def file_exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
