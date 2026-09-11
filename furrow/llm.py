from __future__ import annotations

import asyncio
import os
from pathlib import Path
import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


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
        if not hasattr(self, "_ollama"):
            self._ollama = httpx.AsyncClient(timeout=self.settings.llm_timeout)
        return self._ollama

    async def aclose(self) -> None:
        if hasattr(self, "_ollama") and self._ollama is not None:
            await self._ollama.aclose()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        max_attempts = self.settings.max_attempts
        base_delay = 1.0
        last_exception: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                if self.settings.provider == Provider.ANTHROPIC:
                    return await asyncio.wait_for(
                        self._complete_anthropic(prompt, system, model),
                        timeout=self.settings.llm_timeout,
                    )
                elif self.settings.provider == Provider.OPENAI:
                    return await asyncio.wait_for(
                        self._complete_openai(prompt, system, model),
                        timeout=self.settings.llm_timeout,
                    )
                elif self.settings.provider == Provider.OLLAMA:
                    return await asyncio.wait_for(
                        self._complete_ollama(prompt, system, model),
                        timeout=self.settings.llm_timeout,
                    )
                else:
                    raise ValueError(f"Unsupported provider: {self.settings.provider}")
            except asyncio.CancelledError:
                raise
            except ValueError:
                raise
            except (asyncio.TimeoutError, openai.APIError, anthropic.APIError, httpx.HTTPError) as exc:
                last_exception = exc
                if attempt < max_attempts:
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                continue

        raise RuntimeError(
            f"LLM request failed after {max_attempts} attempts"
        ) from last_exception

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            text = response.content[0].text
        except (IndexError, AttributeError) as exc:
            raise ValueError("Anthropic response is missing text content") from exc
        if not text:
            raise ValueError("Anthropic response text content is empty")
        return text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        try:
            content = response.choices[0].message.content
        except (IndexError, AttributeError) as exc:
            raise ValueError("OpenAI response is missing message content") from exc
        if not content:
            raise ValueError("OpenAI response message content is empty")
        return content

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = await self.ollama.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except Exception as exc:
            raise ValueError("Ollama response is not valid JSON") from exc

        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Ollama response is missing 'message.content'") from exc

        if not content:
            raise ValueError("Ollama response contains empty message content")

        return content

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
