from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic, APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI, APIError as OpenAIAPIError, RateLimitError as OpenAIRateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_turns: int = 5,
    ) -> str:
        """Complete with tool calling support."""
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_tools(prompt, system, model, tools, max_turns)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_with_tools(prompt, system, model, tools, max_turns)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(
        retry=retry_if_exception_type((AnthropicAPIError, AnthropicRateLimitError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
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
        retry=retry_if_exception_type((OpenAIAPIError, OpenAIRateLimitError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
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

    async def _complete_anthropic_with_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_turns: int,
    ) -> str:
        if not tools:
            return await self._complete_anthropic(prompt, system, model)

        # Convert tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            })

        messages = [{"role": "user", "content": prompt}]

        for turn in range(max_turns):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant with access to file system tools.",
                messages=messages,
                tools=anthropic_tools,
            )

            # Check if tool use is requested
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses:
                # No tool use, return the text response
                text_content = [block.text for block in response.content if block.type == "text"]
                return "\n".join(text_content)

            # Add assistant message with tool use
            messages.append({"role": "assistant", "content": response.content})

            # Execute tools and collect results
            tool_results = []
            for tool_use in tool_uses:
                result = await self._execute_tool(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

            # Add tool results as user message
            messages.append({"role": "user", "content": tool_results})

        # Max turns reached, return last response
        text_content = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_content)

    async def _complete_openai_with_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_turns: int,
    ) -> str:
        if not tools:
            return await self._complete_openai(prompt, system, model)

        messages = [
            {"role": "system", "content": system or "You are a helpful coding assistant with access to file system tools."},
            {"role": "user", "content": prompt},
        ]

        for turn in range(max_turns):
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            message = response.choices[0].message
            if not message.tool_calls:
                return message.content or ""

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            })

            # Execute tools and collect results
            for tool_call in message.tool_calls:
                result = await self._execute_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return message.content or ""

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        try:
            if name == "read_file":
                path = args.get("path", "")
                content = await self.read_file(path)
                return f"File content:\n{content}"
            elif name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                await self.write_file(path, content)
                return f"Successfully wrote to {path}"
            elif name == "list_files":
                directory = args.get("directory", ".")
                files = self.list_files(directory)
                return f"Files in {directory}:\n" + "\n".join(files) if files else f"No files in {directory}"
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

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


# Tool definitions for the worker agent
WORKER_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files in a directory recursively",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to list files from"},
            },
            "required": ["directory"],
        },
    },
]