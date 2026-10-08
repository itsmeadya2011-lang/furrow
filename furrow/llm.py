from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


class LLMClient:
    """Thin async wrapper around the configured LLM provider.

    Supports Anthropic, OpenAI, and Ollama. Also exposes a small tool-calling
    API used by the worker agent to read/write files directly.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
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
            self._openai = AsyncOpenAI(api_key=api_key, base_url=self.settings.ollama_base_url)
        return self._openai

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Complete a prompt, optionally with tool calling.

        When ``tools`` is provided the model may emit tool calls; this method
        executes them (file read/write) and returns the final text response.
        """
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, tools, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, tools, max_tokens)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_openai(prompt, system, model, tools, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system or "You are a helpful coding assistant.",
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            kwargs["tools"] = tools
        response = await self.anthropic.messages.create(**kwargs)
        return _extract_text(response.content)

    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        response = await self.openai.chat.completions.create(**kwargs)
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
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]


def _extract_text(blocks: list[Any]) -> str:
    """Pull the concatenated text out of an Anthropic content block list.

    Anthropic responses can contain TextBlock, ThinkingBlock, ToolUseBlock,
    etc. Only TextBlock carries ``.text``; we skip everything else.
    """
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Build an Anthropic-style tool definition dict."""
    return {
        "name": name,
        "description": description,
        "input_schema": parameters,
    }


def openai_tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-style tool definition dict."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


FILE_TOOLS: list[dict[str, Any]] = [
    openai_tool_schema(
        "read_file",
        "Read the contents of a file relative to the workspace.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path relative to workspace root"}},
            "required": ["path"],
        },
    ),
    openai_tool_schema(
        "write_file",
        "Write content to a file, creating parent directories as needed.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace root"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
    ),
]