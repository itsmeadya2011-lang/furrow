from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings
from furrow.config import settings as _default_settings

logger = structlog.get_logger()


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else _default_settings
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
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        logger.debug(
            "llm.complete",
            provider=provider,
            model=model,
            prompt_len=len(prompt),
            system_len=len(system),
        )
        if provider == Provider.ANTHROPIC:
            fn = self._complete_anthropic
        elif provider == Provider.OPENAI:
            fn = self._complete_openai
        elif provider == Provider.OLLAMA:
            fn = self._complete_ollama
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        retries = max(1, int(self.settings.max_retries or 1))
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                text = await fn(prompt, system, model)
                logger.debug("llm.complete.ok", attempt=attempt, out_len=len(text))
                return text
            except Exception as exc:  # noqa: BLE001 - intentional retry surface
                last_exc = exc
                logger.warning(
                    "llm.complete.retry",
                    attempt=attempt,
                    retries=retries,
                    error=str(exc),
                )
                if attempt < retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 10))
        assert last_exc is not None
        raise last_exc

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return ""
        # Anthropic messages can contain non-text blocks (tool_use, etc.); only
        # text blocks expose `.text`. Use getattr to stay type-safe and robust.
        first = response.content[0]
        return getattr(first, "text", None) or ""

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
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        timeout = httpx.Timeout(120.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{self.settings.ollama_base_url}/api/chat", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "")

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
