from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from furrow.config import Provider, Settings, settings

logger = structlog.get_logger(__name__)


class LLMClient:
    """Client for LLM interactions with support for Anthropic, OpenAI, and Ollama."""

    def __init__(self, settings: Settings = settings) -> None:
        self.settings = settings
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._http_client: httpx.AsyncClient | None = None

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
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120)
        return self._http_client

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Get a text completion from the LLM."""
        model = model or self.settings.model
        provider = self.settings.provider
        provider_name = provider.value if isinstance(provider, Provider) else str(provider)
        logger.debug("LLM complete", provider=provider_name, model=model)
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._complete_ollama(prompt, system, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _complete_anthropic(self, prompt: str, system: str, model: str) -> str:
        response = await self._anthropic_request(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self._openai_request(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        payload = {
            "model": model or "llama3.1",
            "system": system or "You are a helpful coding assistant.",
            "prompt": prompt,
            "stream": False,
        }
        response = await self._ollama_request(
            url=f"{self.settings.ollama_base_url}/api/generate",
            json=payload,
        )
        data = response.json()
        return data.get("response", "")

    # ------------------------------------------------------------------
    # Retry-wrapped API calls
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def _anthropic_request(self, **kwargs: Any) -> Any:
        try:
            return await self.anthropic.messages.create(**kwargs)
        except asyncio.TimeoutError:
            logger.warning("Anthropic request timed out, retrying")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def _openai_request(self, **kwargs: Any) -> Any:
        try:
            return await self.openai.chat.completions.create(**kwargs)
        except asyncio.TimeoutError:
            logger.warning("OpenAI request timed out, retrying")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def _ollama_request(self, url: str, json: dict[str, Any]) -> httpx.Response:
        try:
            return await self.http_client.post(url, json=json)
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning("Ollama request failed, retrying", error=str(e))
            raise

    # ------------------------------------------------------------------
    # Tool-calling support
    # ------------------------------------------------------------------

    TOOLS: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative path to the file."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file, creating parent directories if needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute or relative path to the file."},
                        "content": {"type": "string", "description": "The text content to write."},
                        "mode": {
                            "type": "string",
                            "description": "Write mode: 'w' (overwrite) or 'a' (append).",
                            "enum": ["w", "a"],
                            "default": "w",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory recursively.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Path to the directory to list."},
                    },
                    "required": ["directory"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Run a shell command and return its output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The shell command to execute."},
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (default 60).",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str | None = None,
        max_iterations: int = 20,
    ) -> str:
        """Run an LLM agent loop with tool calling.

        The LLM can call tools (read_file, write_file, list_files, execute_command).
        The loop continues until the LLM returns text without tool calls or
        max_iterations is reached.
        """
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._anthropic_tool_loop(messages, system, model, max_iterations)
        elif self.settings.provider == Provider.OPENAI:
            return await self._openai_tool_loop(messages, system, model, max_iterations)
        elif self.settings.provider == Provider.OLLAMA:
            return await self._ollama_tool_loop(messages, system, model, max_iterations)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _anthropic_tool_loop(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        max_iterations: int,
    ) -> str:
        msg_payload: list[dict[str, Any]] = list(messages)
        for iteration in range(max_iterations):
            response = await self._anthropic_request(
                model=model,
                max_tokens=8192,
                system=system or "You are a helpful coding assistant.",
                messages=msg_payload,
                tools=self._anthropic_tools(),
            )
            if response.stop_reason == "tool_use" or any(
                content.type == "tool_use" for content in response.content
            ):
                for content in response.content:
                    if content.type == "tool_use":
                        tool_result = await self._execute_tool(content.name, content.input)
                        msg_payload.append(
                            {
                                "role": "assistant",
                                "content": response.content,
                            }
                        )
                        msg_payload.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": content.id,
                                        "content": tool_result,
                                    }
                                ],
                            }
                        )
                # Continue loop to get next response
                continue
            else:
                text = response.content[0].text if response.content else ""
                return text
        return "Max iterations reached."

    def _anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "tool",
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"],
            }
            for tool in self.TOOLS
        ]

    async def _openai_tool_loop(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        max_iterations: int,
    ) -> str:
        msg_payload: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."}
        ]
        msg_payload.extend(messages)

        for iteration in range(max_iterations):
            response = await self._openai_request(
                model=model,
                messages=msg_payload,
                tools=[{"type": "function", "function": t["function"]} for t in self.TOOLS],
                tool_choice="auto",
            )
            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                msg_payload.append(
                    {"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls}
                )
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    result = await self._execute_tool(func_name, args)
                    msg_payload.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )
                continue
            else:
                return msg.content or ""
        return "Max iterations reached."

    async def _ollama_tool_loop(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        max_iterations: int,
    ) -> str:
        """Ollama tool loop — uses chat format with tool support (ollama >= 0.3)."""
        msg_payload: list[dict[str, Any]] = []
        if system:
            msg_payload.append({"role": "system", "content": system})
        msg_payload.extend(messages)

        for iteration in range(max_iterations):
            payload = {
                "model": model or "llama3.1",
                "messages": msg_payload,
                "stream": False,
                "tools": [t["function"] for t in self.TOOLS],
            }
            response = await self._ollama_request(
                url=f"{self.settings.ollama_base_url}/api/chat",
                json=payload,
            )
            data = response.json()
            msg = data.get("message", {})

            if msg.get("tool_calls"):
                msg_payload.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": msg["tool_calls"]})
                for tool_call in msg["tool_calls"]:
                    func_name = tool_call.get("name", "")
                    args = tool_call.get("arguments", {})
                    result = await self._execute_tool(func_name, args)
                    msg_payload.append({"role": "tool", "content": result})
                continue
            else:
                return msg.get("content", "")
        return "Max iterations reached."

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name with given arguments."""
        logger.debug("Executing tool", name=name, args=args)
        try:
            if name == "read_file":
                path = args["path"]
                content = await self.read_file(path)
                return content
            elif name == "write_file":
                path = args["path"]
                content = args["content"]
                mode = args.get("mode", "w")
                await self.write_file(path, content, mode=mode)
                return f"File written: {path}"
            elif name == "list_files":
                directory = args["directory"]
                files = self.list_files(directory)
                return "\n".join(files) if files else "(empty)"
            elif name == "execute_command":
                command = args["command"]
                timeout = args.get("timeout", 60)
                return await self.execute_command(command, timeout=timeout)
            else:
                return f"Unknown tool: {name}"
        except FileNotFoundError as e:
            return f"Error: File not found: {e}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # File and command utilities
    # ------------------------------------------------------------------

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str, mode: str = "w") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, mode) as f:
            await f.write(content)

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]

    async def execute_command(self, command: str, timeout: float = 60) -> str:
        """Execute a shell command and return stdout+stderr."""
        logger.debug("Executing shell command", command=command, timeout=timeout)
        try:
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(command),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return f"Command timed out after {timeout}s"
            result = stdout.decode() + stderr.decode()
            return result
        except FileNotFoundError as e:
            return f"Error: {e}"

    async def close(self) -> None:
        """Clean up HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
