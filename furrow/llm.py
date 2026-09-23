from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
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

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def complete_with_tools(
        self,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None
    ) -> Any:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_tools(messages, model, tools)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_with_tools(messages, model, tools)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

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

    async def _complete_anthropic_with_tools(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None
    ) -> Any:
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # System is separate in Anthropic
            elif msg["role"] == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg["content"]
                        }
                    ]
                })
            elif msg["role"] == "assistant":
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": json.loads(tc.function.arguments)
                        })
                anthropic_messages.append({"role": "assistant", "content": content})
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), "You are a helpful coding assistant.")

        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=anthropic_messages,
            tools=tools or []
        )

        # Convert to OpenAI-like format for consistency
        class ToolCall:
            def __init__(self, id: str, name: str, arguments: str):
                self.id = id
                self.function = type('Function', (), {'name': name, 'arguments': arguments})()

        class Response:
            def __init__(self, content: str, tool_calls: list = None):
                self.content = content
                self.tool_calls = tool_calls or []

        tool_calls = []
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(block.id, block.name, json.dumps(block.input)))

        return Response(text_content, tool_calls)

    async def _complete_openai_with_tools(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None
    ) -> Any:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or [],
            tool_choice="auto"
        )
        return response.choices[0].message

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