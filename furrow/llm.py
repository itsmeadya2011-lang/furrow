from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        logger.debug("llm.complete.anthropic", model=model)
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        logger.debug("llm.complete.openai", model=model)
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        ollama_model = self.settings.ollama_model
        logger.debug("llm.complete.ollama", model=ollama_model)
        response = await self.ollama.chat.completions.create(
            model=ollama_model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def complete_with_tools(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_with_tools_anthropic(system, messages, tools, model)
        elif self.settings.provider in (Provider.OPENAI, Provider.OLLAMA):
            return await self._complete_with_tools_openai(system, messages, tools, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _complete_with_tools_anthropic(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str
    ) -> tuple[str, list[dict[str, Any]]]:
        logger.debug("llm.tools.anthropic", model=model, tool_count=len(tools))
        anthropic_messages = self._convert_messages_to_anthropic(messages)
        anthropic_tools = self._convert_tools_to_anthropic(tools)

        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=anthropic_messages,
            tools=anthropic_tools,
        )

        text = ""
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "arguments": block.input,
                    "id": block.id,
                })
        return text, tool_calls

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _complete_with_tools_openai(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str
    ) -> tuple[str, list[dict[str, Any]]]:
        logger.debug("llm.tools.openai", model=model, tool_count=len(tools))
        if self.settings.provider == Provider.OLLAMA:
            client = self.ollama
            effective_model = self.settings.ollama_model
        else:
            client = self.openai
            effective_model = model
        openai_messages = self._convert_messages_to_openai(system, messages)
        openai_tools = self._convert_tools_to_openai(tools)

        response = await client.chat.completions.create(
            model=effective_model,
            messages=openai_messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        message = response.choices[0].message
        text = message.content or ""
        tool_calls: list[dict[str, Any]] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc.type == "function":
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                        "id": tc.id,
                    })
        return text, tool_calls

    def _convert_messages_to_anthropic(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "user":
                content = [{"type": "text", "text": msg["content"]}]
                result.append({"role": "user", "content": content})
            elif msg["role"] == "assistant":
                content: list[dict[str, Any]] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls", []):
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                result.append({"role": "assistant", "content": content})
            elif msg["role"] == "tool":
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }],
                })
        return result

    def _convert_messages_to_openai(self, system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."}
        ]
        for msg in messages:
            if msg["role"] == "user":
                result.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                openai_msg: dict[str, Any] = {"role": "assistant", "content": msg.get("content", "")}
                if msg.get("tool_calls"):
                    openai_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ]
                result.append(openai_msg)
            elif msg["role"] == "tool":
                result.append({
                    "role": "tool",
                    "content": msg["content"],
                    "tool_call_id": msg["tool_call_id"],
                })
        return result

    def _convert_tools_to_anthropic(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    def _convert_tools_to_openai(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
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

    async def run_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: int | None = None,
    ) -> str:
        timeout = timeout or self.settings.run_command_timeout
        logger.debug("llm.run_command", command=command, cwd=cwd, timeout=timeout)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return stdout.decode() + stderr.decode()
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Command timed out after {timeout}s and was killed."
        except Exception as e:
            return f"Command failed: {e}"

    async def grep_files(self, pattern: str, directory: str | Path) -> str:
        p = Path(directory)
        if not p.exists():
            return ""

        try:
            proc = await asyncio.create_subprocess_exec(
                "rg", "-n", "--", pattern, str(p),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0 and stdout.strip():
                return stdout.decode()
        except FileNotFoundError:
            pass
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass

        regex = re.compile(pattern)
        results: list[str] = []
        for filepath in p.rglob("*"):
            if filepath.is_file():
                try:
                    content = filepath.read_text(errors="ignore")
                    for lineno, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            results.append(f"{filepath}:{lineno}:{line}")
                except Exception:
                    continue
        return "\n".join(results) if results else "No matches found."
