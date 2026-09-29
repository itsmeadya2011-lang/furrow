from __future__ import annotations

import asyncio
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
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

# Transient errors worth retrying (rate limits, connection/timeout issues).
# Auth and validation errors are intentionally excluded: they cannot recover.
RETRIABLE_ERRORS = (
    anthropic.APIError,
    openai.APIError,
    asyncio.TimeoutError,
    httpx.HTTPError,
)


class LLMClient:
    """Async LLM client for Anthropic, OpenAI and Ollama.

    ``complete`` performs a plain text completion. ``complete_with_tools`` runs an
    agentic loop using built-in file tools (read / write / list) so worker agents
    can actually edit files in the workspace.
    """

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._httpx: httpx.AsyncClient | None = None

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

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_rounds: int = 8,
    ) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._chat_with_tools_anthropic(prompt, system, model, max_rounds)
        elif self.settings.provider == Provider.OPENAI:
            return await self._chat_with_tools_openai(prompt, system, model, max_rounds)
        raise ValueError(
            f"Tool-use is not supported for provider: {self.settings.provider}"
        )

    # ------------------------------------------------------------------ #
    # Retried provider calls
    # ------------------------------------------------------------------ #
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RETRIABLE_ERRORS),
        reraise=True,
    )
    async def _anthropic_create(self, **kwargs: Any) -> Any:
        return await self.anthropic.messages.create(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RETRIABLE_ERRORS),
        reraise=True,
    )
    async def _openai_create(self, **kwargs: Any) -> Any:
        return await self.openai.chat.completions.create(**kwargs)

    # ------------------------------------------------------------------ #
    # Plain completions
    # ------------------------------------------------------------------ #
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self._anthropic_create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self._openai_create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RETRIABLE_ERRORS),
        reraise=True,
    )
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": model or "llama3",
            "messages": [
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = await self.httpx_client.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    # ------------------------------------------------------------------ #
    # Tool-use agentic loops
    # ------------------------------------------------------------------ #
    async def _chat_with_tools_anthropic(
        self, prompt: str, system: str, model: str, max_rounds: int
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for _ in range(max_rounds):
            response = await self._anthropic_create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=messages,
                tools=_anthropic_tool_spec(),
            )
            content = response.content
            messages.append({"role": "assistant", "content": content})

            tool_uses = [b for b in content if getattr(b, "type", "") == "tool_use"]
            if not tool_uses:
                text = " ".join(
                    b.text for b in content if getattr(b, "type", "") == "text" and b.text
                )
                return text.strip()

            for tool_use in tool_uses:
                result = await self._execute_tool(tool_use.name, tool_use.input)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": result,
                            }
                        ],
                    }
                )
        return "Reached the maximum number of tool-use rounds without finishing."

    async def _chat_with_tools_openai(
        self, prompt: str, system: str, model: str, max_rounds: int
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        for _ in range(max_rounds):
            response = await self._openai_create(
                model=model,
                messages=messages,
                tools=_openai_tool_spec(),
                tool_choice="auto",
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return (message.content or "").strip()

            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
            messages.append(assistant_msg)
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments or "{}")
                result = await self._execute_tool(tool_call.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        return "Reached the maximum number of tool-use rounds without finishing."

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                return await self.read_file(args["path"])
            if name == "write_file":
                await self.write_file(args["path"], args["content"])
                return f"Wrote file {args['path']}"
            if name == "list_files":
                files = self.list_files(args["directory"])
                return "\n".join(files) if files else "(empty directory)"
            return f"Unknown tool: {name}"
        except Exception as exc:  # noqa: BLE001 - surface tool errors back to the LLM
            return f"Error executing tool '{name}': {exc}"

    # ------------------------------------------------------------------ #
    # File helpers
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


_FILE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file. Always read a file before editing it.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path of the file to read."}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, overwriting any existing content. Parent directories are created automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to write."},
                "content": {"type": "string", "description": "The content to write to the file."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "Recursively list files under a directory. Returns a newline-separated list of relative paths.",
        "parameters": {
            "type": "object",
            "properties": {"directory": {"type": "string", "description": "Directory to list."}},
            "required": ["directory"],
        },
    },
]


def _anthropic_tool_spec() -> list[dict[str, Any]]:
    return [
        {"type": "tool", "name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in _FILE_TOOLS
    ]


def _openai_tool_spec() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in _FILE_TOOLS
    ]
