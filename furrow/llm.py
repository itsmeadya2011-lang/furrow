from __future__ import annotations

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

    async def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_handlers: dict,
        system: str = "",
        model: str | None = None,
        max_iterations: int = 15,
    ) -> str:
        """Run an LLM loop with tool calling. tools is a list of tool definitions.
        tool_handlers maps tool name -> async callable(handler_input: dict) -> str.
        Loops until the LLM returns a final text answer (no more tool calls)."""
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_with_tools_anthropic(
                prompt, tools, tool_handlers, system, model, max_iterations
            )
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_with_tools_openai(
                prompt, tools, tool_handlers, system, model, max_iterations
            )
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_with_tools_anthropic(
        self,
        prompt: str,
        tools: list[dict],
        tool_handlers: dict,
        system: str,
        model: str,
        max_iterations: int,
    ) -> str:
        anthropic_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ]
        messages: list[dict] = [{"role": "user", "content": prompt}]
        final_text = ""
        for _ in range(max_iterations):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=messages,
                tools=anthropic_tools,
            )
            if response.stop_reason == "tool_use":
                tool_results: list[dict] = []
                for block in response.content:
                    if block.type == "tool_use":
                        handler = tool_handlers.get(block.name)
                        if handler is None:
                            result = f"Tool '{block.name}' is not available"
                        else:
                            result = await handler(block.input)
                        tool_results.append(
                            {"type": "tool_result", "tool_use_id": block.id, "content": result}
                        )
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue
            text_parts = [block.text for block in response.content if block.type == "text"]
            if text_parts:
                final_text = "".join(text_parts)
            return final_text
        return final_text

    async def _complete_with_tools_openai(
        self,
        prompt: str,
        tools: list[dict],
        tool_handlers: dict,
        system: str,
        model: str,
        max_iterations: int,
    ) -> str:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        messages: list[dict] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        final_text = ""
        for _ in range(max_iterations):
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            if tool_calls:
                tool_results: list[dict] = []
                for tool_call in tool_calls:
                    handler = tool_handlers.get(tool_call.function.name)
                    if handler is None:
                        result = f"Tool '{tool_call.function.name}' is not available"
                    else:
                        args = json.loads(tool_call.function.arguments or "{}")
                        result = await handler(args)
                    tool_results.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                    )
                messages.append(
                    {"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls}
                )
                messages.extend(tool_results)
                continue
            final_text = message.content or ""
            return final_text
        return final_text

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
