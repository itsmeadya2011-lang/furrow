from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

log = structlog.get_logger()


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: Any = None
        self._openai: Any = None

    @property
    def anthropic(self) -> Any:
        if self._anthropic is None:
            from anthropic import AsyncAnthropic
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> Any:
        if self._openai is None:
            from openai import AsyncOpenAI
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        log.debug("llm_complete_started", provider=provider, model=model, prompt_length=len(prompt))
        try:
            if provider == Provider.ANTHROPIC:
                result = await self._complete_anthropic(prompt, system, model)
            elif provider == Provider.OPENAI:
                result = await self._complete_openai(prompt, system, model)
            elif provider == Provider.OLLAMA:
                result = await self._complete_ollama(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            log.debug("llm_complete_finished", provider=provider, model=model, response_length=len(result))
            return result
        except Exception as e:
            log.error("llm_complete_failed", provider=provider, model=model, error=str(e))
            raise

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
        import httpx

        base_url = self.settings.ollama_base_url.rstrip("/")
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()
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
