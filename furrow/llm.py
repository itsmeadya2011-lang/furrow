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
from tenacity import retry, stop_after_attempt, wait_exponential

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

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        tools: list[dict] | None = None,
        model: str | None = None,
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_tools(prompt, system, tools, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_with_tools(prompt, system, tools, model)
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

    async def _complete_anthropic_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict] | None,
        model: str,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        while True:
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant with file editing capabilities.",
                messages=messages,
                tools=tools or [],
            )
            
            # Handle tool calls
            tool_results = []
            for content in response.content:
                if content.type == "text":
                    tool_results.append(content.text)
                elif content.type == "tool_use":
                    tool_result = await self._execute_tool(content.name, content.input)
                    tool_results.append(f"Tool {content.name} result: {tool_result}")
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": tool_result
                            }
                        ]
                    })
            
            if all(c.type == "text" for c in response.content):
                return "\n".join(tool_results)

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_openai_with_tools(
        self,
        prompt: str,
        system: str,
        tools: list[dict] | None,
        model: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system or "You are a helpful coding assistant with file editing capabilities."},
            {"role": "user", "content": prompt}
        ]
        
        # Convert tools to OpenAI format
        openai_tools = []
        if tools:
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                })
        
        while True:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools or None,
                tool_choice="auto" if openai_tools else None,
            )
            
            message = response.choices[0].message
            messages.append(message)
            
            if not message.tool_calls:
                return message.content or ""
            
            for tool_call in message.tool_calls:
                tool_result = await self._execute_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

    async def _execute_tool(self, name: str, args: dict) -> str:
        if name == "read_file":
            return await self.read_file(args["path"])
        elif name == "write_file":
            await self.write_file(args["path"], args["content"])
            return f"Successfully wrote to {args['path']}"
        elif name == "list_files":
            files = self.list_files(args["directory"])
            return "\n".join(files)
        else:
            return f"Unknown tool: {name}"

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
