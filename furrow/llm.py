from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from furrow.config import Provider, Settings, settings


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

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.RequestError, AnthropicRateLimitError, OpenAIRateLimitError)),
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None, json_mode: bool = False) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, json_mode)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, json_mode)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, json_mode)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str, json_mode: bool = False) -> str:
        messages = [{"role": "user", "content": prompt}]
        if json_mode:
            # For Anthropic, we prepend JSON instruction to the prompt
            messages[0]["content"] = prompt + "\n\nReturn only valid JSON."
        
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=messages,
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str, json_mode: bool = False) -> str:
        messages = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        
        kwargs = {
            "model": model,
            "messages": messages,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await self.openai.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str, json_mode: bool = False) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt + ("\n\nReturn only valid JSON." if json_mode else "")})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        
        response = await self.ollama.post("/api/chat", json=payload)
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

    async def close(self) -> None:
        if self._ollama_client:
            await self._ollama_client.aclose()
            self._ollama_client = None
