from __future__ import annotations

import json
import os
from pathlib import Path

import aiofiles
import anthropic as anthropic_module
import httpx
import openai as openai_module
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from furrow.config import Provider, Settings, settings
from furrow.logging import get_logger

logger = get_logger("llm")


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

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((anthropic_module.RateLimitError, openai_module.RateLimitError, Exception))
    )
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        model = model or self.settings.model
        logger.debug("complete_request", provider=self.settings.provider.value, model=model, prompt_length=len(prompt))
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
        text = response.content[0].text
        logger.debug("complete_response", provider="anthropic", model=model, response_length=len(text))
        return text

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        logger.debug("complete_response", provider="openai", model=model, response_length=len(text))
        return text

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = f"{self.settings.ollama_base_url}/api/chat"
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system or "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            })
            response.raise_for_status()
            data = response.json()
            text = data.get("message", {}).get("content", "")
            logger.debug("complete_response", provider="ollama", model=model, response_length=len(text))
            return text

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((anthropic_module.RateLimitError, openai_module.RateLimitError, Exception))
    )
    async def complete_with_tools(self, messages: list[dict], model: str | None = None) -> dict:
        """Complete with tool calling support. Returns either tool call dict or final response."""
        model = model or self.settings.model
        logger.debug("complete_with_tools_request", provider=self.settings.provider.value, model=model, message_count=len(messages))
        
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic_with_tools(messages, model)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai_with_tools(messages, model)
        elif self.settings.provider == Provider.OLLAMA:
            # Ollama doesn't have native tool calling yet, fall back to parsing
            return await self._complete_ollama_with_tools(messages, model)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic_with_tools(self, messages: list[dict], model: str) -> dict:
        # Convert messages to Anthropic format
        system = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Define tools for Anthropic
        tools = [
            {
                "name": "read_file",
                "description": "Read a file's contents",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write content to a file (creates directories)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "edit_file",
                "description": "Replace exact string in file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"}
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            },
            {
                "name": "glob_files",
                "description": "Find files matching a glob pattern",
                "input_schema": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"]
                }
            },
            {
                "name": "grep_files",
                "description": "Search file contents with regex",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "include": {"type": "string"}
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "list_files",
                "description": "List all files in a directory",
                "input_schema": {
                    "type": "object",
                    "properties": {"directory": {"type": "string"}},
                    "required": ["directory"]
                }
            },
            {
                "name": "run_command",
                "description": "Run a shell command and return output",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "args": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["cmd"]
                }
            }
        ]
        
        response = await self.anthropic.messages.create(
            model=model,
            max_tokens=4096,
            system=system or "You are a helpful coding assistant.",
            messages=anthropic_messages,
            tools=tools,
            tool_choice={"type": "auto"}
        )
        
        # Check for tool use
        for content in response.content:
            if content.type == "tool_use":
                logger.debug("tool_call", provider="anthropic", tool=content.name)
                return {"tool": content.name, "args": content.input}
        
        # No tool use, return final content
        text = "".join(c.text for c in response.content if c.type == "text")
        logger.debug("complete_response", provider="anthropic", model=model, response_length=len(text))
        return {"content": text}

    async def _complete_openai_with_tools(self, messages: list[dict], model: str) -> dict:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file's contents",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (creates directories)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Replace exact string in file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"}
                        },
                        "required": ["path", "old_string", "new_string"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "glob_files",
                    "description": "Find files matching a glob pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "grep_files",
                    "description": "Search file contents with regex",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "include": {"type": "string"}
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List all files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {"directory": {"type": "string"}},
                        "required": ["directory"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command and return output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                            "args": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["cmd"]
                    }
                }
            }
        ]
        
        response = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            logger.debug("tool_call", provider="openai", tool=tc.function.name)
            return {"tool": tc.function.name, "args": json.loads(tc.function.arguments)}
        
        text = msg.content or ""
        logger.debug("complete_response", provider="openai", model=model, response_length=len(text))
        return {"content": text}

    async def _complete_ollama_with_tools(self, messages: list[dict], model: str) -> dict:
        # For Ollama, we'll use a simpler approach - inject tool descriptions into prompt
        # and parse the response for tool calls
        tool_prompt = """You have access to tools. To use a tool, respond with JSON:
{
  "tool": "tool_name",
  "args": {...}
}

Tools:
- read_file(path): Read a file
- write_file(path, content): Write a file
- edit_file(path, old_string, new_string): Edit a file
- glob_files(pattern): Find files
- grep_files(pattern, include): Search files
- list_files(directory): List files
- run_command(cmd, args): Run command

When done, respond with plain text summary."""
        
        # Add tool prompt to system
        new_messages = messages.copy()
        if new_messages and new_messages[0]["role"] == "system":
            new_messages[0]["content"] += "\n\n" + tool_prompt
        else:
            new_messages.insert(0, {"role": "system", "content": tool_prompt})
        
        response = await self._complete_ollama(
            "\n\n".join(m["content"] for m in new_messages if m["role"] != "system"),
            new_messages[0]["content"] if new_messages[0]["role"] == "system" else "",
            model
        )
        
        # Try to parse tool call from response
        try:
            data = json.loads(response)
            if "tool" in data:
                logger.debug("tool_call", provider="ollama", tool=data["tool"])
                return data
        except json.JSONDecodeError:
            pass
        
        logger.debug("complete_response", provider="ollama", model=model, response_length=len(response))
        return {"content": response}

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
