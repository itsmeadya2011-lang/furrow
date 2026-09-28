from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)

# Default tools available to LLM agents (file + command operations).
DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write."},
                "content": {"type": "string", "description": "Content to write to the file."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files in a directory recursively.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path to list."},
            },
            "required": ["directory"],
        },
    },
    {
        "name": "execute_command",
        "description": "Execute a shell command and return stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
            },
            "required": ["command"],
        },
    },
    {
        "name": "glob_files",
        "description": "Find files matching a glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py'."},
                "directory": {"type": "string", "description": "Directory to search in."},
            },
            "required": ["pattern", "directory"],
        },
    },
    {
        "name": "grep_search",
        "description": "Search for a pattern in file contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Regex or literal search query."},
                "directory": {"type": "string", "description": "Directory to search in."},
                "case_insensitive": {"type": "boolean", "description": "Case insensitive search."},
            },
            "required": ["query", "directory"],
        },
    },
]


# Exceptions that are worth retrying (transient network/server errors).
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.TransportError,
    anthropic.AnthropicError,
    openai.APIError,
    TimeoutError,
)


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None

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
        """Ollama is OpenAI-compatible, so reuse AsyncOpenAI with a custom base URL."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=httpx.Timeout(120),
            )
        if self._openai is None:
            self._openai = AsyncOpenAI(
                api_key="ollama",  # Ollama doesn't require a real API key
                http_client=self._http_client,
                base_url=self.settings.ollama_base_url,
            )
        return self._openai

    # ------------------------------------------------------------------ #
    # Simple completion (no tool-use loop)
    # ------------------------------------------------------------------ #

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        before=before_log(logger, log_level=10),
        after=after_log(logger, log_level=20),
        reraise=True,
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("completing", provider=self.settings.provider, model=model)

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
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

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------ #
    # Tool-use completion (iterative tool calling)
    # ------------------------------------------------------------------ #

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_iterations: int = 10,
    ) -> str:
        """
        Execute an iterative tool-use loop with the LLM.

        The LLM can call tools defined in *tools* (defaults to DEFAULT_TOOLS).
        Each tool call is executed by the corresponding handler in
        ``self._tool_handlers``.  The loop continues until the LLM makes no
        further tool calls or *max_iterations* is reached.
        """
        model = model or self.settings.worker_model
        tools = tools or DEFAULT_TOOLS
        logger.info("starting tool-use loop", model=model, max_iterations=max_iterations)

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._anthropic_tool_loop(prompt, system, model, tools, max_iterations)
        else:
            return await self._openai_tool_loop(prompt, system, model, tools, max_iterations)

    # -- Anthropic tool loop --------------------------------------------

    async def _anthropic_tool_loop(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
        max_iterations: int,
    ) -> str:
        anthropic_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                },
            }
            for t in tools
        ]

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        full_response: list[str] = []

        for iteration in range(max_iterations):
            response = await self._anthropic_create(
                system=system, model=model, messages=messages, tools=anthropic_tools
            )

            tool_results: list[dict[str, Any]] = []
            has_tool_calls = False

            for block in response.content:
                if block.type == "text":
                    full_response.append(block.text)
                elif block.type == "tool_use":
                    has_tool_calls = True
                    tool_name = block.name
                    tool_args = block.input
                    tool_result = await self._execute_tool(tool_name, tool_args)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result,
                        }
                    )

            if not has_tool_calls:
                break

            # Append the assistant message (with tool use) and tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "".join(full_response)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    )
    async def _anthropic_create(self, system: str, model: str, messages, tools) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": system or "You are a helpful coding assistant.",
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return await self.anthropic.messages.create(**kwargs)

    # -- OpenAI-compatible tool loop ------------------------------------

    async def _openai_tool_loop(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
        max_iterations: int,
    ) -> str:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        full_response: list[str] = []

        for iteration in range(max_iterations):
            response = await self._openai_create(model=model, messages=messages, tools=openai_tools)
            msg = response.choices[0].message

            if msg.content:
                full_response.append(msg.content)

            if not msg.tool_calls:
                break

            # Execute all tool calls and collect results
            tool_results: list[dict[str, Any]] = []
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_result = await self._execute_tool(tool_name, tool_args)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )

            # Add assistant message + tool results to conversation
            messages.append(msg.model_dump(exclude_unset=True))
            messages.extend(tool_results)

        return "".join(full_response)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    )
    async def _openai_create(self, model: str, messages, tools) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return await self.openai.chat.completions.create(**kwargs)

    # ------------------------------------------------------------------ #
    # Tool handlers
    # ------------------------------------------------------------------ #

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Dispatch a tool call to the appropriate handler."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Unknown tool: {name}"
        try:
            result = await handler(**args)
            return str(result)
        except Exception as e:  # noqa: BLE001
            logger.error("tool execution failed", tool=name, error=str(e))
            return f"Tool error: {e}"

    async def _tool_read_file(self, path: str) -> str:
        return await self.read_file(path)

    async def _tool_write_file(self, path: str, content: str) -> str:
        await self.write_file(path, content)
        return f"File written: {path}"

    async def _tool_list_files(self, directory: str) -> str:
        return "\n".join(self.list_files(directory))

    async def _tool_execute_command(self, command: str) -> str:
        import asyncio

        try:
            proc = await asyncio.create_subprocess_exec(
                *command.split(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.settings.workspace,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode() + stderr.decode()
            if proc.returncode != 0:
                output += f"\n[exit code: {proc.returncode}]"
            return output
        except asyncio.TimeoutError:
            return "Command timed out."
        except Exception as e:  # noqa: BLE001
            return f"Command error: {e}"

    async def _tool_glob_files(self, pattern: str, directory: str) -> str:
        import fnmatch

        p = Path(directory)
        matches = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, pattern)]
        return "\n".join(matches)

    async def _tool_grep_search(
        self, query: str, directory: str, case_insensitive: bool = False
    ) -> str:
        import re

        p = Path(directory)
        flags = re.IGNORECASE if case_insensitive else 0
        pattern = re.compile(query, flags)
        results: list[str] = []
        for f in p.rglob("*"):
            if f.is_file():
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(content.splitlines(), 1):
                        if pattern.search(line):
                            results.append(f"{f.relative_to(p)}:{i}: {line}")
                except Exception:  # noqa: BLE001, S112
                    continue
        return "\n".join(results) if results else "No matches found."

    # ------------------------------------------------------------------ #
    # File / workspace utilities (unchanged public API)
    # ------------------------------------------------------------------ #

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

