from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings


class LLMClient:
    """Client for interacting with multiple LLM providers.

    Supports Anthropic, OpenAI, and Ollama providers with a unified
    async interface for completions and file I/O.
    """

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None

    @property
    def anthropic(self) -> AsyncAnthropic:
        """Lazy-loaded Anthropic client."""
        if self._anthropic is None:
            api_key = self.settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            self._anthropic = AsyncAnthropic(api_key=api_key)
        return self._anthropic

    @property
    def openai(self) -> AsyncOpenAI:
        """Lazy-loaded OpenAI client."""
        if self._openai is None:
            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            anthropic.APIError,
            anthropic.APIConnectionError,
            openai.APIError,
            openai.APIConnectionError,
        )),
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Get a text completion from the configured LLM provider.

        Args:
            prompt: The user prompt to complete.
            system: Optional system message to steer the model behavior.
            model: Optional model override; falls back to settings.model.

        Returns:
            The generated text response.

        Raises:
            ValueError: If the configured provider is unsupported.
        """
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            anthropic.APIError,
            anthropic.APIConnectionError,
            openai.APIError,
            openai.APIConnectionError,
        )),
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            anthropic.APIError,
            anthropic.APIConnectionError,
            openai.APIError,
            openai.APIConnectionError,
        )),
    )
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
        """Complete a prompt using a local Ollama instance."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system or "You are a helpful coding assistant.",
                    "stream": False,
                },
                timeout=httpx.Timeout(120.0),
            )
            response.raise_for_status()
            return response.json()["response"]

    async def read_file(self, path: str | Path) -> str:
        """Read the contents of a file asynchronously."""
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write content to a file, creating parent directories if needed."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        """List all files recursively under a directory."""
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove leading/trailing markdown code fences from text.

        Handles ```json and ``` variants.
        """
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Complete a prompt and parse the response as JSON.

        Strips markdown code fences before parsing.

        Args:
            prompt: The user prompt requesting JSON output.
            system: Optional system message.
            model: Optional model override.

        Returns:
            Parsed JSON as a dictionary.
        """
        text = await self.complete(prompt, system=system, model=model)
        text = self._strip_code_fence(text)
        return json.loads(text)
