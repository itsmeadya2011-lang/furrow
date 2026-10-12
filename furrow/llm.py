from __future__ import annotations

import json
import os
from pathlib import Path

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


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
        import httpx
        url = f"{self.settings.ollama_base_url}/api/chat"
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system or "You are a helpful coding assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: list[dict] | None = None,
        max_rounds: int = 10,
    ) -> str:
        """Complete with tool calling support."""
        model = model or self.settings.model
        messages: list[dict] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_rounds):
            if self.settings.provider == Provider.ANTHROPIC:
                response = await self.anthropic.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system or "You are a helpful coding assistant.",
                    messages=messages,
                    tools=tools or [],
                )
                content = response.content
                text_parts = []
                tool_uses = []
                for block in content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_uses.append(block)
                messages.append({"role": "assistant", "content": content})
                if not tool_uses:
                    return "\n".join(text_parts)
                # Execute tools
                tool_results = []
                for tool_use in tool_uses:
                    result = await self._execute_tool(tool_use.name, tool_use.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

            elif self.settings.provider == Provider.OPENAI:
                response = await self.openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools or [],
                )
                message = response.choices[0].message
                messages.append(message.model_dump())
                if not message.tool_calls:
                    return message.content or ""
                # Execute tools
                for tool_call in message.tool_calls:
                    result = await self._execute_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

            else:
                # For OLLAMA and others, fall back to simple completion
                return await self.complete(prompt, system, model)

        return "Max tool rounds reached"

    async def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "read_file":
                return await self.read_file(args["path"])
            elif name == "write_file":
                await self.write_file(args["path"], args["content"])
                return f"Successfully wrote to {args['path']}"
            elif name == "list_files":
                files = self.list_files(args["directory"])
                return json.dumps(files)
            else:
                return f"Unknown tool: {name}"
        except Exception as e:  # noqa: BLE001
            return f"Error executing {name}: {e}"

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
