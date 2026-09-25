from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore[import-untyped]
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from furrow.config import Provider, Settings, settings

# Type alias for generic message format used by both providers
MessageDict = dict[str, Any]


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
        messages: list[MessageDict],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | dict[str, Any]:
        """Complete with tool calling support. Returns either a string or a dict with tool_calls."""
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
        # Handle different content block types
        content_block = response.content[0]
        if isinstance(content_block, TextBlock):
            return content_block.text
        return str(content_block)

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
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> str | dict[str, Any]:
        """Complete with tools for Anthropic."""
        # Convert messages to Anthropic format
        system_msg = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": msg["content"],
                        }
                    ],
                })
            else:
                content = msg.get("content", "")
                if msg.get("tool_calls"):
                    # Assistant message with tool calls
                    tool_uses = []
                    for tc in msg["tool_calls"]:
                        tool_uses.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": json.loads(tc["function"]["arguments"]),
                        })
                    anthropic_messages.append({"role": "assistant", "content": tool_uses})
                else:
                    anthropic_messages.append({"role": msg["role"], "content": content})

        # Convert tools to Anthropic format
        anthropic_tools = []
        if tools:
            for tool in tools:
                if tool["type"] == "function":
                    f = tool["function"]
                    anthropic_tools.append({
                        "name": f["name"],
                        "description": f["description"],
                        "input_schema": f["parameters"],
                    })

        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system_msg or "You are a helpful coding assistant.",
            messages=anthropic_messages,  # type: ignore[arg-type]
            tools=anthropic_tools if anthropic_tools else None,  # type: ignore[arg-type]
        )

        # Check for tool use
        tool_calls = []
        text_content = ""
        for content in response.content:
            if content.type == "text":
                text_content += content.text
            elif content.type == "tool_use":
                tool_calls.append({
                    "id": content.id,
                    "function": {
                        "name": content.name,
                        "arguments": json.dumps(content.input),
                    },
                })

        if tool_calls:
            return {"tool_calls": tool_calls, "content": text_content}
        return text_content

    async def _complete_openai_with_tools(
        self,
        messages: list[MessageDict],
        model: str,
        tools: list[dict[str, Any]] | None,
    ) -> str | dict[str, Any]:
        """Complete with tools for OpenAI."""
        # Convert messages to OpenAI format
        openai_messages: list[ChatCompletionMessageParam] = messages  # type: ignore[assignment]
        # Convert tools to OpenAI format
        openai_tools = []
        if tools:
            for tool in tools:
                if tool["type"] == "function":
                    openai_tools.append(tool)

        response = await self.openai.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=openai_messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None,
        )

        message = response.choices[0].message
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            return {"tool_calls": tool_calls, "content": message.content or ""}
        return message.content or ""

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