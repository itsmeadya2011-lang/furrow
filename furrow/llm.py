from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast

import aiofiles
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, ToolUseBlock
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from furrow.config import Provider, Settings
from furrow.config import settings as _default_settings

DEFAULT_SYSTEM = "You are a helpful coding assistant."


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


FURROW_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file at the given path, "
                "creating parent directories as needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write."},
                    "content": {"type": "string", "description": "Content to write to the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files recursively in the given directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to list files in."},
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command and return its stdout and stderr output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                },
                "required": ["command"],
            },
        },
    },
]


def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool definitions to Anthropic-format."""
    return [
        {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "input_schema": tool["function"]["parameters"],
        }
        for tool in tools
    ]


def _content_to_text(content: Any) -> str:
    """Extract plain text from an OpenAI-style message content field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", "") or "")
                # image_url and other block types carry no plain text here.
        return "".join(parts)
    return str(content)


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-style messages to Anthropic-style messages.

    Returns a tuple of (system_string, anthropic_messages). System messages are
    collapsed into a single system string; tool results are mapped to user
    messages carrying ``tool_result`` content blocks.
    """
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            text = _content_to_text(content)
            if text:
                system_parts.append(text)
            continue

        if role == "user":
            text = _content_to_text(content)
            anthropic_messages.append({"role": "user", "content": text})
            continue

        if role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            text = _content_to_text(content)
            if text:
                content_blocks.append({"type": "text", "text": text})
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    raw_args = func.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    else:
                        args = raw_args or {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": name,
                            "input": args,
                        }
                    )
            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})
            anthropic_messages.append({"role": "assistant", "content": content_blocks})
            continue

        if role == "tool":
            text = _content_to_text(content)
            tool_call_id = msg.get("tool_call_id", "")
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "output": text,
                        },
                    ],
                },
            )
            continue

        # Fallback for any unexpected roles: coerce to a user message.
        text = _content_to_text(content)
        anthropic_messages.append({"role": role, "content": text})

    system = "\n\n".join(system_parts) if system_parts else ""
    return system, anthropic_messages


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or _default_settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._ollama: AsyncOpenAI | None = None

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
    def ollama(self) -> AsyncOpenAI:
        if self._ollama is None:
            base_url = self.settings.ollama_base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
            self._ollama = AsyncOpenAI(base_url=base_url, api_key="ollama")
        return self._ollama

    def _openai_compatible_client(self) -> AsyncOpenAI:
        if self.settings.provider == Provider.OLLAMA:
            return self.ollama
        return self.openai

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._complete_openai(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or DEFAULT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return cast(TextBlock, response.content[0]).text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self._openai_compatible_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def complete_with_tools(
        self,
        prompt: str = "",
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> ToolResponse:
        model = model or self.settings.model
        if tools is None:
            tools = FURROW_TOOLS
        if messages is None:
            messages = [
                {"role": "system", "content": system or DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_with_tools_anthropic(messages, model, tools)
        elif self.settings.provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._complete_with_tools_openai(messages, model, tools)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_with_tools_anthropic(
        self, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]]
    ) -> ToolResponse:
        system, anthropic_messages = _to_anthropic_messages(messages)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or DEFAULT_SYSTEM,
            messages=cast(Any, anthropic_messages),
            tools=cast(Any, _to_anthropic_tools(tools)),
        )
        text: str | None = None
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if isinstance(block, TextBlock):
                text = (text or "") + block.text
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )
        return ToolResponse(text=text, tool_calls=tool_calls)

    async def _complete_with_tools_openai(
        self, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]]
    ) -> ToolResponse:
        response = await self._openai_compatible_client().chat.completions.create(
            model=model,
            messages=cast(Any, messages),
            tools=cast(Any, tools),
        )
        message = response.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            if tc.type == "function":
                try:
                    arguments: dict[str, Any] = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )
        return ToolResponse(text=message.content, tool_calls=tool_calls)

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "read_file":
            return await self.read_file(arguments["path"])
        elif name == "write_file":
            await self.write_file(arguments["path"], arguments["content"])
            return ""
        elif name == "list_files":
            return "\n".join(self.list_files(arguments["directory"]))
        elif name == "run_command":
            return await self.run_command(arguments["command"])
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            content: str = await f.read()
        return content

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

    async def run_command(self, command: str, timeout: float = 120.0) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"Command timed out after {timeout}s: {command}")
        output = (stdout or b"") + (stderr or b"")
        return output.decode("utf-8", errors="replace")
