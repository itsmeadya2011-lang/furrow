from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiofiles
import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from furrow.config import Provider, Settings, settings


# Tool definitions for worker agents
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file (creates or overwrites)",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace specific text in a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "old_string": {"type": "string", "description": "The exact text to replace"},
                "new_string": {"type": "string", "description": "The replacement text"}
            },
            "required": ["path", "old_string", "new_string"]
        }
    },
    {
        "name": "bash",
        "description": "Execute a shell command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"}
            },
            "required": ["command"]
        }
    },
]


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

    async def complete_with_tools(self, messages: list[dict[str, Any]], model: str | None = None) -> Any:
        """Execute an LLM call with tool calling support."""
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_tools(messages, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_with_tools(messages, model)
        elif self.settings.provider == Provider.OLLAMA:
            # Ollama doesn't support structured output in the same way
            # Fall back to regular completion
            prompt = "\n".join(m[-1]["content"] for m in messages if isinstance(m[-1]["content"], str))
            result = await self._complete_ollama(prompt, "", model)
            # Create a mock response for compatibility
            class MockContent:
                def __init__(self, text):
                    self.type = "text"
                    self.text = text
            class MockResponse:
                def __init__(self, content):
                    self.content = [MockContent(content)]
            return MockResponse(result)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic_with_tools(self, messages: list[dict[str, Any]], model: str) -> Any:
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = msg["role"] if msg["role"] != "assistant" else "assistant"
            content = msg["content"]
            anthropic_messages.append({"role": role, "content": content})
        
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system="You are a helpful coding assistant that uses tools to complete tasks.",
            messages=anthropic_messages,
            tools=TOOLS,
        )
        return response

    async def _complete_openai_with_tools(self, messages: list[dict[str, Any]], model: str) -> Any:
        # Convert to OpenAI format with tool definitions
        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=[
                {"type": "function", "function": tool}
                for tool in TOOLS
            ],
            tool_choice="auto",
        )
        return response

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
        """Complete using Ollama local models."""
        import httpx
        base_url = self.settings.ollama_base_url or "http://localhost:11434"
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system or "You are a helpful coding assistant.",
                    "stream": False,
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

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
