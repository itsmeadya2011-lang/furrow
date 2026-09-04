from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class CompletionResult:
    text: str
    tool_calls: list[ToolCall]


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

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> str | CompletionResult:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, tools)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, tools)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(
        self, prompt: str, system: str, model: str, tools: list[dict] | None
    ) -> str | CompletionResult:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
        )
        text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        if tools is None:
            return text
        tool_calls = [
            ToolCall(name=block.name, arguments=block.input)
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]
        return CompletionResult(text=text, tool_calls=tool_calls)

    async def _complete_openai(
        self, prompt: str, system: str, model: str, tools: list[dict] | None
    ) -> str | CompletionResult:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            tools=tools,
        )
        text = response.choices[0].message.content or ""
        if tools is None:
            return text
        raw_tool_calls = response.choices[0].message.tool_calls or []
        tool_calls = [
            ToolCall(
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in raw_tool_calls
        ]
        return CompletionResult(text=text, tool_calls=tool_calls)

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
