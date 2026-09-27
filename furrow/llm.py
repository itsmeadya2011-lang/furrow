from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


class LLMClient:
    TOOLS: list[dict[str, Any]] = [
        {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to read."},
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file at the given path, creating parent directories if needed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to write to."},
                    "content": {"type": "string", "description": "The content to write."},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "list_files",
            "description": "List all files in a directory recursively.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "The directory to list files in."},
                },
                "required": ["directory"],
            },
        },
        {
            "name": "run_command",
            "description": "Run a shell command and return combined stdout and stderr.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                },
                "required": ["command"],
            },
        },
        {
            "name": "grep_files",
            "description": "Search for a regex pattern in a directory using ripgrep (rg).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The regex pattern to search for."},
                    "directory": {
                        "type": "string",
                        "description": "The directory to search in.",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
        },
    ]

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
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_iterations: int = 20,
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_tools(prompt, system, model, max_iterations)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_with_tools(prompt, system, model, max_iterations)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama_with_tools(prompt, system, model, max_iterations)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    def _format_tools_for_provider(self, provider: Provider) -> list[dict[str, Any]]:
        if provider == Provider.ANTHROPIC:
            return [
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }
                for tool in self.TOOLS
            ]
        elif provider == Provider.OPENAI:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in self.TOOLS
            ]
        elif provider == Provider.OLLAMA:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in self.TOOLS
            ]
        return []

    async def _call_tool(self, name: str, **kwargs: Any) -> tuple[str, bool]:
        method = getattr(self, name, None)
        if not callable(method):
            return f"Unknown tool: {name}", True
        try:
            result = await method(**kwargs)
            if isinstance(result, str):
                return result, False
            return str(result), False
        except Exception as e:  # noqa: BLE001
            return f"Error executing tool '{name}': {e}", True

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
        model: str,
        max_iterations: int,
    ) -> str:
        tools = self._format_tools_for_provider(Provider.ANTHROPIC)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        last_text = ""
        for _ in range(max_iterations):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=messages,
                tools=tools,
            )
            assistant_content: list[dict[str, Any]] = []
            tool_uses: list[Any] = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                    last_text += block.text
                elif block.type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                    tool_uses.append(block)
            messages.append({"role": "assistant", "content": assistant_content})
            if not tool_uses:
                return last_text
            for tool_use in tool_uses:
                result, is_error = await self._call_tool(tool_use.name, **tool_use.input)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": result,
                                "is_error": is_error,
                            }
                        ],
                    }
                )
        return last_text

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
        model: str,
        max_iterations: int,
    ) -> str:
        tools = self._format_tools_for_provider(Provider.OPENAI)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        last_text = ""
        for _ in range(max_iterations):
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )
            message = response.choices[0].message
            if message.content:
                last_text = message.content
            assistant_message: dict[str, Any] = {"role": "assistant"}
            if message.content:
                assistant_message["content"] = message.content
            else:
                assistant_message["content"] = ""
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_message)
            if not message.tool_calls:
                return message.content or ""
            for tool_call in message.tool_calls:
                arguments = (
                    json.loads(tool_call.function.arguments)
                    if tool_call.function.arguments
                    else {}
                )
                result, _is_error = await self._call_tool(tool_call.function.name, **arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
        return last_text

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        import httpx

        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system or "You are a helpful coding assistant."
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return data.get("response", "")

    async def _complete_ollama_with_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        max_iterations: int,
    ) -> str:
        import httpx

        tools = self._format_tools_for_provider(Provider.OLLAMA)
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        last_text = ""
        for _ in range(max_iterations):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.settings.ollama_base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "tools": tools,
                    },
                )
                response.raise_for_status()
                data = response.json()
            message = data.get("message", {})
            content = message.get("content", "")
            last_text = content
            tool_calls = message.get("tool_calls") or []
            messages.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
                if tool_calls
                else {"role": "assistant", "content": content}
            )
            if not tool_calls:
                return content
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                arguments_str = func.get("arguments", "")
                arguments = json.loads(arguments_str) if arguments_str else {}
                result, _is_error = await self._call_tool(name, **arguments)
                messages.append({"role": "tool", "content": result, "name": name})
        return last_text

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        return f"File written successfully: {p}"

    async def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    async def run_command(self, command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "Command timed out after 120 seconds."
        return (stdout.decode() if stdout else "") + (stderr.decode() if stderr else "")

    async def grep_files(self, pattern: str, directory: str | Path = ".") -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "rg",
                pattern,
                str(directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return "Error: ripgrep (rg) is not installed."
        stdout, stderr = await proc.communicate()
        output = (stdout.decode() if stdout else "") + (stderr.decode() if stderr else "")
        return output
