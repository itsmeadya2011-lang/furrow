from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


@dataclass
class ToolCall:
    """A single tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of a turn: optional text plus any tool calls requested."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


# File-system tools the LLM can invoke. Kept provider-neutral; each is described
# with JSON Schema so both Anthropic and OpenAI can bind them.
FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read and return the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "File contents to write.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "list_files",
        "description": "List all files under a directory recursively.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to list (defaults to workspace root).",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


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

    async def run_tool_loop(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_rounds: int = 10,
    ) -> str:
        """Run a tool-use loop: call the LLM, execute any tool calls it requests,
        feed the results back, and repeat until the LLM returns plain text."""
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._anthropic_tool_loop(
                prompt, system, model, tools, max_rounds
            )
        elif self.settings.provider == Provider.OPENAI:
            return await self._openai_tool_loop(prompt, system, model, tools, max_rounds)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _anthropic_tool_loop(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_rounds: int,
    ) -> str:
        system_text = system or "You are a helpful coding assistant."
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for _ in range(max_rounds):
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": 4096,
                "system": system_text,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "required"
            response = await self.anthropic.messages.create(**kwargs)

            text_parts: list[str] = []
            tool_calls: list[Any] = []
            assistant_blocks: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    assistant_blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_calls.append(block)
                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": dict(block.input),
                        }
                    )
            messages.append({"role": "assistant", "content": assistant_blocks})

            if not tool_calls:
                return "".join(text_parts) or "Task complete."

            tool_result_blocks: list[dict[str, Any]] = []
            for call in tool_calls:
                try:
                    output = await self.execute_tool(call.name, dict(call.input))
                except Exception as exc:  # noqa: BLE001 - surface to the LLM
                    output = f"Error: {exc}"
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": output,
                    }
                )
            messages.append({"role": "user", "content": tool_result_blocks})

        return "Max tool rounds reached."

    async def _openai_tool_loop(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]] | None,
        max_rounds: int,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system or "You are a helpful coding assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        for _ in range(max_rounds):
            kwargs: dict[str, Any] = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "required"
            response = await self.openai.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in (msg.tool_calls or [])
            ]
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": assistant_tool_calls or None,
                }
            )
            # Remove the tool_calls key entirely when None so the payload stays clean.
            if not assistant_tool_calls:
                messages[-1].pop("tool_calls", None)
                return msg.content or "Task complete."

            for tc in (msg.tool_calls or []):
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                    output = await self.execute_tool(tc.function.name, arguments)
                except Exception as exc:  # noqa: BLE001 - surface to the LLM
                    output = f"Error: {exc}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )

        return "Max tool rounds reached."

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

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a single tool call by name and return a string result."""
        if name == "read_file":
            return await self.read_file(arguments["path"])
        if name == "write_file":
            await self.write_file(arguments["path"], arguments["content"])
            return f"Wrote {arguments['path']}"
        if name == "list_files":
            return json.dumps(self.list_files(arguments.get("directory", ".")))
        raise ValueError(f"Unknown tool: {name}")
