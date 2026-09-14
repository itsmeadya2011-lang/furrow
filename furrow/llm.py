from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

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
        system: str = "",
        model: str | None = None,
        tools: list[dict] | None = None,
        tool_executor: Callable | None = None,
        max_tool_rounds: int = 10,
    ) -> str:
        """
        Complete a prompt with tool-use support.

        Args:
            prompt: The user prompt
            system: System message
            model: Model override
            tools: List of tool schemas (provider-specific format)
            tool_executor: Async callable that takes (tool_name, tool_input) and returns output string
            max_tool_rounds: Maximum number of tool-use rounds before giving up

        Returns:
            Final text response after all tool calls are resolved
        """
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_with_tools_anthropic(
                prompt, system, model, tools, tool_executor, max_tool_rounds
            )
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_with_tools_openai(
                prompt, system, model, tools, tool_executor, max_tool_rounds
            )
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_with_tools_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict] | None,
        tool_executor: Callable | None,
        max_tool_rounds: int,
    ) -> str:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=messages,
            tools=tools or [],
        )
        rounds = 0
        while self._anthropic_has_tool_use(response):
            if rounds >= max_tool_rounds:
                break
            if tool_executor is None:
                raise ValueError("tool_executor is required when tools are requested")
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await tool_executor(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=messages,
                tools=tools or [],
            )
            rounds += 1
        return self._anthropic_text(response)

    @staticmethod
    def _anthropic_has_tool_use(response: Any) -> bool:
        return any(block.type == "tool_use" for block in response.content)

    @staticmethod
    def _anthropic_text(response: Any) -> str:
        for block in response.content:
            if block.type == "text":
                return block.text or ""
        return ""

    async def _complete_with_tools_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict] | None,
        tool_executor: Callable | None,
        max_tool_rounds: int,
    ) -> str:
        messages: list[dict] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or [],
        )
        rounds = 0
        while self._openai_has_tool_calls(response):
            if rounds >= max_tool_rounds:
                break
            if tool_executor is None:
                raise ValueError("tool_executor is required when tools are requested")
            tool_calls = response.choices[0].message.tool_calls or []
            messages.append(
                {
                    "role": "assistant",
                    "content": response.choices[0].message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                result = await tool_executor(tc.function.name, tc.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools or [],
            )
            rounds += 1
        return response.choices[0].message.content or ""

    @staticmethod
    def _openai_has_tool_calls(response: Any) -> bool:
        return bool(response.choices[0].message.tool_calls)

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
