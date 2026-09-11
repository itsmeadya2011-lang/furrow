from __future__ import annotations

import asyncio
import fnmatch
import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None
        self._tools: list[dict[str, Any]] = [
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
                "description": "Write content to a file",
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
                "description": "List files in a directory",
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
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"}
                    },
                    "required": ["command"],
                },
            },
        ]

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
    def httpx_client(self) -> httpx.AsyncClient:
        if self._httpx is None:
            self._httpx = httpx.AsyncClient(timeout=60.0)
        return self._httpx

    @property
    def _openai_tools(self) -> list[dict[str, Any]]:
        """Convert internal tool definitions to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in self._tools
        ]

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
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = await self.httpx_client.post(
            f"{self.settings.ollama_base_url}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    async def run_command(
        self, command: str, cwd: str | Path | None = None
    ) -> tuple[str, str, int]:
        if cwd is None:
            cwd = self.settings.workspace
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return stdout.decode(), stderr.decode(), proc.returncode or 0

    async def complete_with_tools(
        self, prompt: str, system: str = "", model: str | None = None, max_iterations: int = 10
    ) -> str:
        model = model or self.settings.model
        system_prompt = system or "You are a helpful coding assistant."

        # Anthropic takes system as a separate parameter, not in messages.
        # OpenAI expects system as the first message in the array.
        if self.settings.provider == Provider.ANTHROPIC:
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        for _ in range(max_iterations):
            if self.settings.provider == Provider.ANTHROPIC:
                response = await self.anthropic.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=self._tools,
                )
                content = response.content
                tool_calls = [c for c in content if c.type == "tool_use"]
                if not tool_calls:
                    return "".join(c.text for c in content if c.type == "text")
                messages.append({"role": "assistant", "content": content})
                for tool_call in tool_calls:
                    result = await self._execute_tool(tool_call.name, tool_call.input)
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_call.id,
                                    "content": result,
                                }
                            ],
                        }
                    )
            elif self.settings.provider == Provider.OPENAI:
                response = await self.openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=self._openai_tools,
                )
                message = response.choices[0].message
                tool_calls = message.tool_calls or []
                if not tool_calls:
                    return message.content or ""
                messages.append(message.model_dump())
                for tool_call in tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    result = await self._execute_tool(tool_call.function.name, args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
            elif self.settings.provider == Provider.OLLAMA:
                return await self._complete_ollama(prompt, system, model)
            else:
                raise ValueError(f"Unsupported provider: {self.settings.provider}")

        return "Max iterations reached without final response"

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                return await self.read_file(args["path"])
            elif name == "write_file":
                await self.write_file(args["path"], args["content"])
                return f"Successfully wrote to {args['path']}"
            elif name == "list_files":
                return "\n".join(self.list_files(args["directory"]))
            elif name == "run_command":
                stdout, stderr, code = await self.run_command(args["command"])
                return f"stdout:\n{stdout}\nstderr:\n{stderr}\nreturncode: {code}"
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error executing {name}: {e}"

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    def list_files(self, directory: str | Path, exclude: set[str] | None = None) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        exclude = exclude or {"__pycache__", "*.pyc", ".git", "node_modules", ".venv", "venv"}
        results: list[str] = []
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            rel_parts = f.relative_to(p).parts
            if any(fnmatch.fnmatch(part, pat) for part in rel_parts for pat in exclude):
                continue
            results.append(str(f.relative_to(p)))
        return results