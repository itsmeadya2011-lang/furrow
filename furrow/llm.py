from __future__ import annotations

import json
import os
import subprocess
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

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"}
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file, creating directories if needed",
                "input_schema": {
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
                "description": "List files in a directory recursively",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory to list files from"}
                    },
                    "required": ["directory"],
                },
            },
            {
                "name": "run_command",
                "description": "Run a shell command and return its output",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
                    },
                    "required": ["command"],
                },
            },
        ]

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "read_file":
            try:
                content = await self.read_file(arguments["path"])
                return content
            except Exception as e:
                return f"Error reading file: {e}"
        elif name == "write_file":
            try:
                await self.write_file(arguments["path"], arguments["content"])
                return f"Successfully wrote to {arguments['path']}"
            except Exception as e:
                return f"Error writing file: {e}"
        elif name == "list_files":
            try:
                files = self.list_files(arguments.get("directory", "."))
                return "\n".join(files) if files else "No files found."
            except Exception as e:
                return f"Error listing files: {e}"
        elif name == "run_command":
            timeout = arguments.get("timeout", 120)
            try:
                result = subprocess.run(
                    arguments["command"],
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                output = result.stdout + result.stderr
                return output if output else f"Command completed with return code {result.returncode}"
            except subprocess.TimeoutExpired:
                return f"Command timed out after {timeout}s"
            except Exception as e:
                return f"Error running command: {e}"
        else:
            return f"Unknown tool: {name}"

    async def complete(self, prompt: str, system: str = "", model: str | None = None, use_tools: bool = True) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, use_tools)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, use_tools)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str, use_tools: bool) -> str:
        messages = [{"role": "user", "content": prompt}]
        tools = self.tools if use_tools else []

        while True:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": 4096,
                "system": system or "You are a helpful coding assistant.",
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
            response = await self.anthropic.messages.create(**kwargs)

            tool_use_blocks = [block for block in response.content if getattr(block, "type", None) == "tool_use"]
            if not tool_use_blocks:
                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                return "".join(text_parts)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_use_blocks:
                result = await self._execute_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            messages.append({"role": "user", "content": tool_results})

    async def _complete_openai(self, prompt: str, system: str, model: str, use_tools: bool) -> str:
        messages = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        tool_schemas = []
        if use_tools:
            for t in self.tools:
                tool_schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": t["input_schema"],
                        },
                    }
                )

        while True:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas
                kwargs["tool_choice"] = "auto"
            response = await self.openai.chat.completions.create(**kwargs)
            message = response.choices[0].message

            tool_calls = message.tool_calls
            if not tool_calls:
                return message.content or ""

            messages.append(message)
            for tool_call in tool_calls:
                result = await self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments),
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

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
