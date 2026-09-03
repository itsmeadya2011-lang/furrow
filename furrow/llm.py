from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: Optional[str] = None


@dataclass
class LlmResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def _llm_retry() -> Any:
    """Tenacity retry decorator honoring the configured retry_attempts."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(settings.retry_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )


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
                api_key="ollama",
                base_url=self.settings.ollama_base_url,
            )
        return self._ollama

    # ------------------------------------------------------------------ complete
    @_llm_retry()
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        response = await self.chat(
            messages=[
                {"role": "user", "content": prompt},
            ],
            system=system or "You are a helpful coding assistant.",
            model=model,
        )
        return response.text

    # --------------------------------------------------------------------- chat
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        system: str = "",
    ) -> LlmResponse:
        provider = self.settings.provider
        if provider == Provider.ANTHROPIC:
            return await self._chat_anthropic(messages, tools, model, system)
        if provider == Provider.OPENAI:
            return await self._chat_openai(messages, tools, model, system)
        if provider == Provider.OLLAMA:
            return await self._chat_ollama(messages, tools, model, system)
        raise ValueError(f"Unsupported provider: {provider}")

    @_llm_retry()
    async def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        system: str,
    ) -> LlmResponse:
        model = model or self.settings.model
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("params", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]
        kwargs: dict = {
            "model": model,
            "max_tokens": 4096,
            "system": system or "You are a helpful coding assistant.",
            "messages": messages,
        }
        if anthropic_tools is not None:
            kwargs["tools"] = anthropic_tools
        response = await self.anthropic.messages.create(**kwargs)
        return self._parse_anthropic(response)

    @staticmethod
    def _parse_anthropic(response: Any) -> LlmResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text" or hasattr(block, "text"):
                text_parts.append(getattr(block, "text", "") or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=getattr(block, "name", ""),
                        arguments=dict(getattr(block, "input", None) or {}),
                        id=getattr(block, "id", None),
                    )
                )
        return LlmResponse(text="".join(text_parts), tool_calls=tool_calls)

    @_llm_retry()
    async def _chat_openai(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        system: str,
    ) -> LlmResponse:
        model = model or self.settings.model
        full_messages = list(messages)
        if system:
            full_messages = [{"role": "system", "content": system}] + full_messages
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("params", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]
        kwargs: dict = {
            "model": model,
            "messages": full_messages,
        }
        if openai_tools is not None:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"
        response = await self.openai.chat.completions.create(**kwargs)
        return self._parse_openai(response)

    @staticmethod
    def _parse_openai(response: Any) -> LlmResponse:
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        tool_calls: list[ToolCall] = []
        for c in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(c.function.arguments) if c.function.arguments else {}
            except (TypeError, ValueError):
                args = {}
            tool_calls.append(
                ToolCall(
                    name=c.function.name,
                    arguments=args if isinstance(args, dict) else {},
                    id=c.id,
                )
            )
        return LlmResponse(text=text, tool_calls=tool_calls)

    @_llm_retry()
    async def _chat_ollama(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        system: str,
    ) -> LlmResponse:
        model = model or "llama3"
        full_messages = list(messages)
        if system:
            full_messages = [{"role": "system", "content": system}] + full_messages
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("params", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]
        kwargs: dict = {
            "model": model,
            "messages": full_messages,
        }
        if openai_tools is not None:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"
        response = await self.ollama.chat.completions.create(**kwargs)
        return self._parse_openai(response)

    # ----------------------------------------------------------- file helpers
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
