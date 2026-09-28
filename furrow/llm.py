from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    APIStatusError as AnthropicAPIStatusError,
    AsyncAnthropic,
)
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    APIStatusError as OpenAIAPIStatusError,
    AsyncOpenAI,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings
from furrow.tools import ToolRegistry

logger = structlog.get_logger(__name__)

_ANTHROPIC_RETRY_ERRORS = (AnthropicAPIStatusError, AnthropicAPIConnectionError)
_OPENAI_RETRY_ERRORS = (OpenAIAPIStatusError, OpenAIAPIConnectionError)


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
            base_url = self.settings.ollama_base_url.rstrip("/") + "/v1"
            self._ollama = AsyncOpenAI(
                api_key="ollama",
                base_url=base_url,
            )
        return self._ollama

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
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def complete_with_tools(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        tools: ToolRegistry | None = None,
        max_iterations: int = 10,
    ) -> str:
        model = model or self.settings.model
        if tools is None:
            tools = ToolRegistry()
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_tools(
                prompt, system, model, tools, max_iterations
            )
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_with_tools(
                prompt, system, model, tools, max_iterations
            )
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama_with_tools(
                prompt, system, model, tools, max_iterations
            )
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_ANTHROPIC_RETRY_ERRORS),
        reraise=True,
    )
    async def _complete_anthropic(
        self, prompt: str, system: str, model: str
    ) -> str:
        logger.debug("calling_anthropic", model=model)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        logger.info("anthropic_completion", model=model)
        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_OPENAI_RETRY_ERRORS),
        reraise=True,
    )
    async def _complete_openai(
        self, prompt: str, system: str, model: str
    ) -> str:
        logger.debug("calling_openai", model=model)
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system or "You are a helpful coding assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        logger.info("openai_completion", model=model)
        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_OPENAI_RETRY_ERRORS),
        reraise=True,
    )
    async def _complete_ollama(
        self, prompt: str, system: str, model: str
    ) -> str:
        logger.debug("calling_ollama", model=model)
        response = await self.ollama.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system or "You are a helpful coding assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        logger.info("ollama_completion", model=model)
        return text

    async def _complete_anthropic_with_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: ToolRegistry,
        max_iterations: int,
    ) -> str:
        tool_specs = tools.get_anthropic_tools()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]

        for _ in range(max_iterations):
            response = await self.anthropic.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are a helpful coding assistant.",
                messages=messages,
                tools=tool_specs,
            )

            text_parts: list[str] = []
            tool_results: list[dict[str, Any]] = []
            has_tool_calls = False

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text or "")
                elif block.type == "tool_use":
                    has_tool_calls = True
                    tool_name = block.name
                    tool_input = block.input
                    logger.debug(
                        "anthropic_tool_call", tool=tool_name, input=tool_input
                    )
                    try:
                        result = await tools.execute(
                            tool_name, **tool_input
                        )
                    except Exception as e:
                        result = f"Error executing tool {tool_name}: {e}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            if has_tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            b.to_dict() for b in response.content
                        ],
                    }
                )
                messages.append(
                    {"role": "user", "content": tool_results}
                )
            else:
                return "\n".join(text_parts)

        return "\n".join(text_parts) if text_parts else ""

    async def _complete_openai_with_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: ToolRegistry,
        max_iterations: int,
    ) -> str:
        tool_specs = tools.get_openai_tools()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system or "You are a helpful coding assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_iterations):
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_specs,
                tool_choice="auto",
            )

            message = response.choices[0].message
            text = message.content or ""
            tool_calls = message.tool_calls

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    tool_name = tc.function.name
                    raw_args = tc.function.arguments or "{}"
                    try:
                        tool_input = (
                            json.loads(raw_args)
                            if isinstance(raw_args, str)
                            else raw_args
                        )
                    except json.JSONDecodeError as e:
                        result = f"Error parsing tool arguments for {tool_name}: {e}"
                    else:
                        logger.debug(
                            "openai_tool_call", tool=tool_name, input=tool_input
                        )
                        try:
                            result = await tools.execute(
                                tool_name, **tool_input
                            )
                        except Exception as e:
                            result = f"Error executing tool {tool_name}: {e}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
            else:
                logger.info("openai_tool_completion", model=model)
                return text

        return ""

    async def _complete_ollama_with_tools(
        self,
        prompt: str,
        system: str,
        model: str,
        tools: ToolRegistry,
        max_iterations: int,
    ) -> str:
        tool_specs = tools.get_openai_tools()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system or "You are a helpful coding assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_iterations):
            response = await self.ollama.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_specs,
                tool_choice="auto",
            )

            message = response.choices[0].message
            text = message.content or ""
            tool_calls = message.tool_calls

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    tool_name = tc.function.name
                    raw_args = tc.function.arguments or "{}"
                    try:
                        tool_input = (
                            json.loads(raw_args)
                            if isinstance(raw_args, str)
                            else raw_args
                        )
                    except json.JSONDecodeError as e:
                        result = f"Error parsing tool arguments for {tool_name}: {e}"
                    else:
                        logger.debug(
                            "ollama_tool_call", tool=tool_name, input=tool_input
                        )
                        try:
                            result = await tools.execute(
                                tool_name, **tool_input
                            )
                        except Exception as e:
                            result = f"Error executing tool {tool_name}: {e}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
            else:
                logger.info("ollama_tool_completion", model=model)
                return text

        return ""

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
