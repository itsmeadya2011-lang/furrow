from __future__ import annotations

import os
from pathlib import Path

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(
                api_key=api_key,
                timeout=self.settings.request_timeout,
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
                timeout=self.settings.request_timeout,
            )
        return self._openai

    @property
    def ollama(self) -> AsyncOpenAI:
        """Ollama speaks the OpenAI-compatible API; the key is ignored but required."""
        if self._ollama is None:
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
                timeout=self.settings.request_timeout,
            )
        return self._ollama

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        if provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        if provider == Provider.OLLAMA:
            return await self._complete_openai(prompt, system, model, client=self.ollama)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Thinking/tool-use blocks may come first and have no `.text`; keep only text blocks.
        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        if not text:
            seen = ", ".join(sorted({str(getattr(b, "type", "?")) for b in response.content}))
            raise ValueError(f"Anthropic model {model!r} returned no text blocks (saw: {seen})")
        return text

    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        client: AsyncOpenAI | None = None,
    ) -> str:
        client = client or self.openai
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

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
        return sorted(str(f.relative_to(p)) for f in p.rglob("*") if f.is_file())
