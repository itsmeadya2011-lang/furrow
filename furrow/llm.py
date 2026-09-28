from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings


FILE_TOOLS_ANTHROPIC = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "The path to the file to read."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path to the file to write."},
                    "content": {"type": "string", "description": "The content to write to the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in a directory (recursively).",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "The directory path to list files from."}},
                "required": ["path"],
            },
        },
    },
]


class LLMClient:
    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
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
                base_url=self.settings.ollama_base_url + "/v1",
                api_key="ollama",
            )
        return self._ollama

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, client=self.openai)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_openai(prompt, system, model, client=self.ollama)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def _complete_openai(
        self, prompt: str, system: str, model: str, client: AsyncOpenAI | None = None
    ) -> str:
        client = client or self.openai
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict] | None = None,
        system: str = "",
        model: str | None = None,
        max_iterations: int = 10,
    ) -> str:
        model = model or self.settings.model
        tools = tools or FILE_TOOLS_ANTHROPIC
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        system_prompt = system or "You are a helpful coding assistant."

        for _ in range(max_iterations):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=tools,
            )
            tool_results: list[dict[str, Any]] = []
            text_response: str | None = None
            for block in response.content:
                if block.type == "text":
                    text_response = block.text
                elif block.type == "tool_use":
                    result = await self._execute_tool(block.name, block.input)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result}
                    )
            if not tool_results:
                return text_response or ""
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        return text_response or ""

    async def _execute_tool(self, name: str, kwargs: dict) -> str:
        if name == "read_file":
            return await self.read_file(kwargs["path"])
        elif name == "write_file":
            await self.write_file(kwargs["path"], kwargs["content"])
            return f"File {kwargs['path']} written successfully."
        elif name == "list_files":
            return "\n".join(self.list_files(kwargs["path"]))
        return f"Unknown tool: {name}"

    async def aclose(self) -> None:
        if self._anthropic is not None:
            await self._anthropic.close()
        if self._openai is not None:
            await self._openai.close()
        if self._ollama is not None:
            await self._ollama.close()

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
