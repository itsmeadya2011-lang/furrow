from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles

from furrow.config import Provider, Settings, settings


class LLMClient:
    """Unified async LLM client supporting Anthropic, OpenAI, and Ollama."""

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: Any | None = None
        self._openai: Any | None = None
        self._ollama: Any | None = None

    # ------------------------------------------------------------------
    # Provider clients (lazy initialization)
    # ------------------------------------------------------------------

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

    @property
    def ollama(self) -> Any:
        if self._ollama is None:
            from ollama import AsyncClient

            self._ollama = AsyncClient(host=self.settings.ollama_base_url)
        return self._ollama

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Single-turn completion. Auto-selects provider from settings."""
        model = model or self.settings.model
        provider = self.settings.provider

        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, temperature, max_tokens)
        elif provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, temperature, max_tokens)
        elif provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(
        self, prompt: str, system: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(
        self, prompt: str, system: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        response = await self.ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["message"]["content"]

    # ------------------------------------------------------------------
    # Conversation (multi-turn) support for agentic loops
    # ------------------------------------------------------------------

    async def converse(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Multi-turn conversation.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}.
            model: Override model name.
        """
        model = model or self.settings.model
        provider = self.settings.provider

        if provider == Provider.ANTHROPIC:
            return await self._converse_anthropic(messages, model, temperature, max_tokens)
        elif provider == Provider.OPENAI:
            return await self._converse_openai(messages, model, temperature, max_tokens)
        elif provider == Provider.OLLAMA:
            return await self._converse_ollama(messages, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _converse_anthropic(
        self, messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int
    ) -> str:
        system = ""
        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "You are a helpful coding assistant.",
            messages=anthropic_messages,
        )
        return response.content[0].text

    async def _converse_openai(
        self, messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int
    ) -> str:
        formatted = [
            {"role": "system" if m["role"] == "system" else m["role"], "content": m["content"]}
            for m in messages
        ]
        response = await self.openai.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=formatted,
        )
        return response.choices[0].message.content or ""

    async def _converse_ollama(
        self, messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int
    ) -> str:
        ollama_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        response = await self.ollama.chat(
            model=model,
            messages=ollama_messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["message"]["content"]

    # ------------------------------------------------------------------
    # JSON response helper (with retry on parse failure)
    # ------------------------------------------------------------------

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        retries: int = 2,
    ) -> dict[str, Any]:
        """Request JSON output, retrying on parse failure.

        On the first attempt, sends the prompt directly.
        On retry, prepends a system instruction telling the LLM to output
        only valid JSON.
        """
        for attempt in range(retries + 1):
            if attempt == 0:
                response = await self.complete(prompt, system, model, temperature, max_tokens)
            else:
                fix_system = "You must output ONLY valid JSON. No markdown, no explanation."
                response = await self.complete(
                    prompt,
                    system=fix_system,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Try one more cleanup: extract the first valid JSON object
                cleaned = self._extract_json_object(response)
                if cleaned is not None:
                    return cleaned
                if attempt >= retries:
                    raise
        # Should never reach here, but for type safety:
        raise ValueError(f"Failed to parse JSON after {retries + 1} attempts: {response}")

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        """Attempt to find and parse the first valid JSON object in text."""
        # Find all candidate substrings that look like JSON objects
        start = text.find("{")
        while start != -1:
            end = text.rfind("}", start + 1)
            if end == -1:
                break
            candidate = text[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            start = text.find("{", start + 1)
        return None

    # ------------------------------------------------------------------
    # File system helpers
    # ------------------------------------------------------------------

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    async def append_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "a") as f:
            await f.write(content)

    async def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    async def file_exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    async def read_file_safe(self, path: str | Path) -> str | None:
        """Read a file, returning None if it doesn't exist."""
        p = Path(path)
        if not p.exists():
            return None
        return await self.read_file(p)
