from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings

FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "write_file",
        "description": (
            "Write or overwrite a file with the given content. Use this to create or "
            "modify project files. Always provide the full file content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "content": {"type": "string", "description": "Full contents of the file"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file to inspect it before editing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
            },
            "required": ["path"],
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
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
            )
        return self._ollama

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
        response = await self.ollama.chat.completions.create(
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
        system: str = "",
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_rounds: int = 5,
    ) -> str:
        """Run an agentic loop where the model can call file tools."""
        model = model or self.settings.model
        tools = tools if tools is not None else FILE_TOOLS
        tool_map = {
            "write_file": self.write_file,
            "read_file": self.read_file,
        }

        if self.settings.provider == Provider.ANTHROPIC:
            return await self._tool_loop_anthropic(prompt, system, model, tools, tool_map, max_rounds)
        elif self.settings.provider in (Provider.OPENAI, Provider.OLLAMA):
            client = self.openai if self.settings.provider == Provider.OPENAI else self.ollama
            return await self._tool_loop_openai(prompt, system, model, tools, tool_map, client, max_rounds)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _tool_loop_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
        tool_map: dict[str, Any],
        max_rounds: int,
    ) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        final_text = ""
        for _ in range(max_rounds):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=messages,
                tools=tools,
            )
            assistant_content: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "text":
                    final_text = block.text
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                    fn = tool_map.get(block.name)
                    if fn is None:
                        result = f"Unknown tool: {block.name}"
                    else:
                        try:
                            if block.name == "read_file":
                                result = await fn(block.input["path"])
                            else:
                                await fn(block.input["path"], block.input["content"])
                                result = f"Wrote {block.input['path']}"
                        except Exception as e:  # noqa: BLE001
                            result = f"Error: {e}"
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result}
                    )
            messages.append({"role": "assistant", "content": assistant_content})
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break
        return final_text

    async def _tool_loop_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[dict[str, Any]],
        tool_map: dict[str, Any],
        client: AsyncOpenAI,
        max_rounds: int,
    ) -> str:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        final_text = ""
        for _ in range(max_rounds):
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
            )
            choice = response.choices[0]
            msg = choice.message
            if msg.content:
                final_text = msg.content
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
            if not msg.tool_calls:
                break
            for call in msg.tool_calls:
                fn = tool_map.get(call.function.name)
                try:
                    args = json.loads(call.function.arguments)
                    if call.function.name == "read_file":
                        result = await fn(args["path"])
                    else:
                        await fn(args["path"], args["content"])
                        result = f"Wrote {args['path']}"
                except Exception as e:  # noqa: BLE001
                    result = f"Error: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )
        return final_text

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
