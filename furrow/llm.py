from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
from anthropic import APIError as AnthropicAPIError
from anthropic import AsyncAnthropic
from openai import APIError as OpenAIAPIError
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    wait_exponential,
)

from furrow.config import Provider, Settings
from furrow.config import settings as _default_settings

_logger = structlog.get_logger(__name__)


def _stop_after_max_retries(retry_state: Any) -> bool:
    instance = retry_state.args[0] if retry_state.args else None
    limit = getattr(instance, "max_retries", 3)
    return retry_state.attempt_number >= limit


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class LLMClient:
    def __init__(self, settings: Settings | None = None, max_retries: int = 3) -> None:
        self.settings = settings if settings is not None else _default_settings
        self.max_retries = max_retries
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

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        provider = self.settings.provider
        _logger.info("complete", provider=provider.value, model=model)
        if provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif provider == Provider.OLLAMA:
            return await self._complete_openai(prompt, system, model, client=self.ollama)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @retry(
        stop=_stop_after_max_retries,
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (AnthropicAPIError, OpenAIAPIError, httpx.HTTPError)
        ),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
        reraise=True,
    )
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    @retry(
        stop=_stop_after_max_retries,
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (AnthropicAPIError, OpenAIAPIError, httpx.HTTPError)
        ),
        before_sleep=before_sleep_log(_logger, logging.WARNING),
        reraise=True,
    )
    async def _complete_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        client: AsyncOpenAI | None = None,
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
        system: str,
        model: str,
        tools: list[ToolSpec],
        handler: Callable[[str, dict[str, Any]], str | Any],
    ) -> str:
        provider = self.settings.provider
        _logger.info("complete_with_tools", provider=provider.value, model=model, tools=len(tools))
        if provider == Provider.ANTHROPIC:
            return await self._complete_with_tools_anthropic(
                prompt, system, model, tools, handler
            )
        elif provider in (Provider.OPENAI, Provider.OLLAMA):
            client = self.ollama if provider == Provider.OLLAMA else self.openai
            return await self._complete_with_tools_openai(
                prompt, system, model, tools, handler, client
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _complete_with_tools_anthropic(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[ToolSpec],
        handler: Callable[[str, dict[str, Any]], str | Any],
    ) -> str:
        native_tools = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        text_parts: list[str] = []
        for _ in range(10):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=messages,  # type: ignore[arg-type]
                tools=native_tools,  # type: ignore[arg-type]
            )
            assistant_content: list[dict[str, Any]] = []
            tool_calls: list[Any] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_calls.append(block)
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
            messages.append({"role": "assistant", "content": assistant_content})
            if not tool_calls:
                return "".join(text_parts)
            for tc in tool_calls:
                result = handler(tc.name, tc.input)
                if inspect.isawaitable(result):
                    result = await result
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": str(result),
                            }
                        ],
                    }
                )
        return "".join(text_parts)

    async def _complete_with_tools_openai(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: list[ToolSpec],
        handler: Callable[[str, dict[str, Any]], str | Any],
        client: AsyncOpenAI,
    ) -> str:
        native_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        text_parts: list[str] = []
        for _ in range(10):
            response = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                tools=native_tools,  # type: ignore[arg-type]
            )
            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []
            function_calls = [tc for tc in tool_calls if tc.type == "function"]
            if msg.content:
                text_parts.append(msg.content)
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if function_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in function_calls
                ]
            messages.append(assistant_msg)
            if not function_calls:
                return "".join(text_parts)
            for tc in function_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                result = handler(tc.function.name, args)
                if inspect.isawaitable(result):
                    result = await result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    }
                )
        return "".join(text_parts)

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as f:
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

    async def run_command(self, command: str) -> str:
        if not self.settings.allow_shell:
            raise PermissionError(
                "Shell execution is disabled (set allow_shell=True to enable)"
            )
        _logger.warning("run_command", command=command)
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (stdout.decode() + stderr.decode()).strip()
