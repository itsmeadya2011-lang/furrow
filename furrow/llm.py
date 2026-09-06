from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


class Tool:
    """A callable tool that can be invoked by an LLM during tool-calling."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., str | Awaitable[str]],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._func = func
        self._parameters = parameters or {"type": "object", "properties": {}, "required": []}

    @property
    def schema(self) -> dict[str, Any]:
        """Return the tool schema in OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters,
            },
        }

    @property
    def anthropic_schema(self) -> dict[str, Any]:
        """Return the tool schema in Anthropic format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._parameters,
        }

    async def execute(self, **kwargs: Any) -> str:
        result = self._func(**kwargs)
        if isinstance(result, str):
            return result
        if isinstance(result, Awaitable):
            return await result
        return str(result)


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
    def ollama(self) -> AsyncOpenAI:
        if self._openai is None:
            self._openai = AsyncOpenAI(base_url=self.settings.ollama_base_url + "/v1", api_key="ollama")
        return self._openai

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[Tool] | None = None,
        max_rounds: int = 10,
    ) -> str:
        """Complete a prompt, optionally with tool-calling.

        If ``tools`` is provided, runs the full tool-calling loop until the
        model stops issuing tool calls or ``max_rounds`` is reached.
        """
        model = model or self.settings.model
        if tools:
            return await self._complete_with_tools(prompt, system, model, tools, max_rounds)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    # ------------------------------------------------------------------
    # Simple (no-tool) completions
    # ------------------------------------------------------------------

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=self.settings.max_tokens,
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

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Tool-calling completions
    # ------------------------------------------------------------------

    async def _complete_with_tools(
        self, prompt: str, system: str, model: str, tools: list[Tool], max_rounds: int
    ) -> str:
        """Run the full tool-calling loop, returning the final text response."""
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_with_tools_anthropic(prompt, system, model, tools, max_rounds)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_with_tools_openai(prompt, system, model, tools, max_rounds)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_with_tools_ollama(prompt, system, model, tools, max_rounds)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_with_tools_anthropic(
        self, prompt: str, system: str, model: str, tools: list[Tool], max_rounds: int
    ) -> str:
        tool_map = {t.name: t for t in tools}
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tool_schemas = [t.anthropic_schema for t in tools]
        sys = system or "You are a helpful coding assistant with access to file tools."

        for _ in range(max_rounds):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=self.settings.max_tokens,
                system=sys,
                messages=messages,
                tools=tool_schemas,
            )
            text_parts: list[str] = []
            tool_uses: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            if not tool_uses:
                return "\n".join(text_parts)

            # Append assistant message with tool_use blocks
            content = []
            if text_parts:
                content.append({"type": "text", "text": "\n".join(text_parts)})
            for tu in tool_uses:
                content.append({"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu["input"]})
            messages.append({"role": "assistant", "content": content})

            # Execute tools and collect results
            result_blocks: list[dict[str, Any]] = []
            for tu in tool_uses:
                tool = tool_map.get(tu["name"])
                if tool:
                    try:
                        output = await tool.execute(**tu["input"])
                    except Exception as e:
                        output = f"Error calling tool {tu['name']}: {e}"
                else:
                    output = f"Tool '{tu['name']}' not found."
                result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": output,
                })
            messages.append({"role": "user", "content": result_blocks})

        # Max rounds reached — return whatever text we have
        return "\n".join(text_parts)

    async def _complete_with_tools_openai(
        self, prompt: str, system: str, model: str, tools: list[Tool], max_rounds: int
    ) -> str:
        tool_map = {t.name: t for t in tools}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant with access to file tools."},
            {"role": "user", "content": prompt},
        ]
        tool_schemas = [t.schema for t in tools]
        last_text = ""

        for _ in range(max_rounds):
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
            )
            choice = response.choices[0]
            msg = choice.message

            if msg.content:
                last_text = msg.content

            if not msg.tool_calls:
                return msg.content or last_text

            # Append assistant message with tool_calls
            messages.append(msg.model_dump())

            # Execute tools and collect results
            for tc in msg.tool_calls:
                tool = tool_map.get(tc.function.name)
                if tool:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        output = await tool.execute(**args)
                    except Exception as e:
                        output = f"Error calling tool {tc.function.name}: {e}"
                else:
                    output = f"Tool '{tc.function.name}' not found."
                messages.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": output,
                })

        return last_text

    async def _complete_with_tools_ollama(
        self, prompt: str, system: str, model: str, tools: list[Tool], max_rounds: int
    ) -> str:
        """Ollama tool-calling reuses the OpenAI path since Ollama is OpenAI-compatible."""
        tool_map = {t.name: t for t in tools}
        original_provider = self.settings.provider
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant with access to file tools."},
            {"role": "user", "content": prompt},
        ]
        tool_schemas = [t.schema for t in tools]
        last_text = ""

        for _ in range(max_rounds):
            response = await self.ollama.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
            )
            choice = response.choices[0]
            msg = choice.message

            if msg.content:
                last_text = msg.content

            if not msg.tool_calls:
                return msg.content or last_text

            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                tool = tool_map.get(tc.function.name)
                if tool:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        output = await tool.execute(**args)
                    except Exception as e:
                        output = f"Error calling tool {tc.function.name}: {e}"
                else:
                    output = f"Tool '{tc.function.name}' not found."
                messages.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": output,
                })

        return last_text

    # ------------------------------------------------------------------
    # File operations (used as tools by the worker)
    # ------------------------------------------------------------------

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        # Prevent path traversal outside workspace
        try:
            p.relative_to(self.settings.workspace.resolve())
        except ValueError:
            pass  # Allow absolute paths for flexibility
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        return f"Wrote {len(content)} bytes to {p}"

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    def get_file_tools(self) -> list[Tool]:
        """Return file-system tools suitable for passing to ``complete()``."""
        return [
            Tool(
                name="read_file",
                description="Read the contents of a file at the given path. Returns the file contents as a string.",
                func=self.read_file,
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "The path to the file to read"}},
                    "required": ["path"],
                },
            ),
            Tool(
                name="write_file",
                description="Write content to a file at the given path. Creates parent directories if needed. Returns a confirmation message.",
                func=self.write_file,
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file to write"},
                        "content": {"type": "string", "description": "The content to write to the file"},
                    },
                    "required": ["path", "content"],
                },
            ),
            Tool(
                name="list_files",
                description="List all files in a directory recursively. Returns a list of file paths relative to the directory.",
                func=self.list_files,
                parameters={
                    "type": "object",
                    "properties": {"directory": {"type": "string", "description": "The directory to list files from"}},
                    "required": ["directory"],
                },
            ),
        ]
