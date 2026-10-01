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
class Tool:
    """A tool the LLM can invoke via chat()."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A single tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Result of a chat() call: optional text and any tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


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

    def _ollama_client(self) -> AsyncOpenAI:
        """Build an OpenAI-compatible client pointed at the local Ollama server."""
        return AsyncOpenAI(base_url=self.settings.ollama_base_url, api_key="ollama")

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_via_openai(self._ollama_client(), prompt, system, model)
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

    async def _complete_via_openai(
        self, client: AsyncOpenAI, prompt: str, system: str, model: str
    ) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        return await self._complete_via_openai(self.openai, prompt, system, model)

    # -- Tool-calling chat --

    async def chat(
        self,
        tools: list[Tool],
        messages: list[dict],
        system: str = "",
        model: str | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Chat with tool support. `messages` use the unified internal format:
        role user/assistant/tool with optional `tool_calls`/`tool_call_id`."""
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._chat_anthropic(tools, messages, system, model, max_tokens)
        elif self.settings.provider == Provider.OPENAI:
            return await self._chat_openai(tools, messages, system, model, max_tokens)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._chat_ollama(tools, messages, system, model, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _chat_openai(
        self, tools: list[Tool], messages: list[dict], system: str, model: str, max_tokens: int
    ) -> LLMResponse:
        return await self._chat_via_openai(self.openai, tools, messages, system, model, max_tokens)

    async def _chat_ollama(
        self, tools: list[Tool], messages: list[dict], system: str, model: str, max_tokens: int
    ) -> LLMResponse:
        return await self._chat_via_openai(
            self._ollama_client(), tools, messages, system, model, max_tokens
        )

    async def _chat_via_openai(
        self,
        client: AsyncOpenAI,
        tools: list[Tool],
        messages: list[dict],
        system: str,
        model: str,
        max_tokens: int,
    ) -> LLMResponse:
        msgs: list[dict] = []
        if system:
            msgs.append(
                {"role": "system", "content": system or "You are a helpful coding assistant."}
            )
        for m in messages:
            role = m.get("role")
            content = m.get("content", "") or ""
            if role == "user":
                msgs.append({"role": "user", "content": content})
            elif role == "assistant":
                msg: dict = {"role": "assistant"}
                msg["content"] = content if content else None
                tool_calls = m.get("tool_calls") or []
                if tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {})),
                            },
                        }
                        for tc in tool_calls
                    ]
                msgs.append(msg)
            elif role == "tool":
                msgs.append(
                    {"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": content}
                )
        funcs = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
            }
            for t in tools
        ]
        response = await client.chat.completions.create(
            model=model, messages=msgs, tools=funcs, max_tokens=max_tokens
        )
        msg_resp = response.choices[0].message
        text = msg_resp.content or ""
        tool_calls: list[ToolCall] = []
        for tc in getattr(msg_resp, "tool_calls", []) or []:
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, Exception):  # noqa: BLE001
                arguments = {"raw": tc.function.arguments}
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
            )
        return LLMResponse(text=text, tool_calls=tool_calls)

    async def _chat_anthropic(
        self, tools: list[Tool], messages: list[dict], system: str, model: str, max_tokens: int
    ) -> LLMResponse:
        anthropic_msgs: list[dict] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "") or ""
            if role == "user":
                anthropic_msgs.append(
                    {"role": "user", "content": [{"type": "text", "text": content}]}
                )
            elif role == "assistant":
                blocks: list[dict] = [{"type": "text", "text": content}] if content else []
                for tc in m.get("tool_calls", []) or []:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc.get("arguments", {}),
                        }
                    )
                anthropic_msgs.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                anthropic_msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id"),
                                "content": content,
                            }
                        ],
                    }
                )
        anthropic_tools = [
            {
                "type": "tool",
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or "You are a helpful coding assistant.",
            messages=anthropic_msgs,
            tools=anthropic_tools,
        )
        text = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text += block.text
            elif btype == "tool_use":
                raw = block.input
                arguments = (
                    dict(raw) if isinstance(raw, dict) else json.loads(raw or "{}")
                )
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=arguments)
                )
        return LLMResponse(text=text, tool_calls=tool_calls)

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
