from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, ClassVar

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as default_settings

logger = structlog.get_logger(__name__)

_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (
            httpx.HTTPError,
            anthropic.APIError,
            openai.APIError,
            asyncio.TimeoutError,
            ConnectionError,
        )
    ),
    reraise=True,
)


class LLMClient:
    _TOOL_DEFINITIONS: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to read."},
            },
            "required": ["path"],
        },
        {
            "name": "write_file",
            "description": "Write content to a file, creating parent directories as needed.",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to write."},
                "content": {"type": "string", "description": "The content to write to the file."},
            },
            "required": ["path", "content"],
        },
        {
            "name": "edit_file",
            "description": "Replace old_string with new_string in a file. Errors on no match or multiple matches.",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to edit."},
                "old_string": {"type": "string", "description": "The string to replace."},
                "new_string": {"type": "string", "description": "The replacement string."},
            },
            "required": ["path", "old_string", "new_string"],
        },
        {
            "name": "list_files",
            "description": "List all files in a directory recursively.",
            "properties": {
                "directory": {"type": "string", "description": "The directory to list files in."},
            },
            "required": ["directory"],
        },
        {
            "name": "run_bash",
            "description": "Execute a shell command and return its stdout and stderr.",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "timeout": {"type": "integer", "description": "Timeout in seconds for the command.", "default": 60},
            },
            "required": ["command"],
        },
    ]

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or default_settings
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
            self._ollama = AsyncOpenAI(
                base_url=self.settings.ollama_base_url, api_key="ollama"
            )
        return self._ollama

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        model = model or self.settings.model
        logger.debug("complete_start", provider=str(self.settings.provider), model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            response = await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            response = await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            response = await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

        if not response or not response.strip():
            raise ValueError("LLM returned an empty response")
        logger.debug(
            "complete_end",
            provider=str(self.settings.provider),
            response_length=len(response),
        )
        return response

    @_RETRY
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else None
        return text or ""

    @_RETRY
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @_RETRY
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def get_tools(self) -> list[dict[str, Any]]:
        if self.settings.provider == Provider.ANTHROPIC:
            return [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": {
                        "type": "object",
                        "properties": tool["properties"],
                        "required": tool["required"],
                    },
                }
                for tool in self._TOOL_DEFINITIONS
            ]
        else:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": {
                            "type": "object",
                            "properties": tool["properties"],
                            "required": tool["required"],
                        },
                    },
                }
                for tool in self._TOOL_DEFINITIONS
            ]

    async def _execute_tool(self, name: str, input_data: dict[str, Any]) -> str:
        if name == "read_file":
            return await self.read_file(input_data["path"])
        elif name == "write_file":
            await self.write_file(input_data["path"], input_data["content"])
            return "File written successfully."
        elif name == "edit_file":
            return await self._edit_file(
                input_data["path"],
                input_data["old_string"],
                input_data["new_string"],
            )
        elif name == "list_files":
            files = self.list_files(input_data["directory"])
            return "\n".join(files) if files else ""
        elif name == "run_bash":
            timeout = input_data.get("timeout", self.settings.llm_timeout)
            try:
                timeout = int(timeout)
            except (TypeError, ValueError):
                timeout = self.settings.llm_timeout
            return await self._run_bash(input_data["command"], timeout)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _edit_file(self, path: str, old_string: str, new_string: str) -> str:
        content = await self.read_file(path)
        if old_string not in content:
            raise ValueError(f"old_string not found in {path}")
        occurrences = content.count(old_string)
        if occurrences > 1:
            raise ValueError(
                f"old_string matches {occurrences} occurrences in {path}; "
                "provide a more specific string"
            )
        await self.write_file(path, content.replace(old_string, new_string))
        return "File edited successfully."

    async def _run_bash(self, command: str, timeout: int) -> str:
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
            return "Command timed out."
        output = (
            stdout.decode(errors="replace") + stderr.decode(errors="replace")
        ).strip()
        return output

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_iterations: int = 10,
    ) -> str:
        model = model or self.settings.model
        tools = self.get_tools()
        system = system or "You are a helpful coding assistant."

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._anthropic_complete_with_tools(
                prompt, system, model, tools, max_iterations
            )
        elif self.settings.provider in (Provider.OPENAI, Provider.OLLAMA):
            client = self.ollama if self.settings.provider == Provider.OLLAMA else self.openai
            return await self._openai_complete_with_tools(
                client, prompt, system, model, tools, max_iterations
            )
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @_RETRY
    async def _anthropic_create_with_tools(
        self, model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        return await self.anthropic.messages.create(
            model=model, max_tokens=4096, system=system, messages=messages, tools=tools
        )

    @_RETRY
    async def _openai_create_with_tools(
        self, client: AsyncOpenAI, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        return await client.chat.completions.create(
            model=model, messages=messages, tools=tools
        )

    async def _anthropic_complete_with_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
        max_iterations: int,
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for _ in range(max_iterations):
            response = await self._anthropic_create_with_tools(
                model, system, messages, tools
            )
            tool_uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                text = "".join(
                    b.text for b in response.content if getattr(b, "type", "") == "text"
                )
                if not text.strip():
                    raise ValueError("LLM returned an empty response")
                return text
            messages.append({"role": "assistant", "content": response.content})
            result_blocks = []
            for block in tool_uses:
                try:
                    result = await self._execute_tool(block.name, block.input)
                except (KeyError, ValueError, OSError, TypeError) as e:
                    result = f"Tool error: {e}"
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        "is_error": False,
                    }
                )
            messages.append({"role": "user", "content": result_blocks})
        raise ValueError("Max iterations reached without completion")

    async def _openai_complete_with_tools(
        self,
        client: AsyncOpenAI,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
        max_iterations: int,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        for _ in range(max_iterations):
            response = await self._openai_create_with_tools(
                client, model, messages, tools
            )
            message = response.choices[0].message
            if not message.tool_calls:
                content = message.content or ""
                if not content.strip():
                    raise ValueError("LLM returned an empty response")
                return content
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls,
                }
            )
            for tc in message.tool_calls:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
                try:
                    result = await self._execute_tool(tc.function.name, arguments)
                except (KeyError, ValueError, OSError, TypeError) as e:
                    result = f"Tool error: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": result,
                    }
                )
        raise ValueError("Max iterations reached without completion")

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
