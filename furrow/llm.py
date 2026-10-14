from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel

from furrow.config import Provider, Settings, settings


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    tool_call: ToolCall
    result: str
    error: str | None = None


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

    def _get_tools(self) -> list[dict[str, Any]]:
        return [
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
                "description": "Write content to a file",
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
                "name": "list_files",
                "description": "List files in a directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory to list files from"}
                    },
                    "required": ["directory"]
                }
            },
            {
                "name": "run_command",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run"},
                        "cwd": {"type": "string", "description": "Working directory"}
                    },
                    "required": ["command"]
                }
            }
        ]

    async def complete(self, prompt: str, system: str = "", model: str | None = None, tools: bool = False) -> str:
        model = model or self.settings.model
        if self.settings.provider == Provider.ANTHROPIC:
            return await self._complete_anthropic(prompt, system, model, tools)
        elif self.settings.provider == Provider.OPENAI:
            return await self._complete_openai(prompt, system, model, tools)
        else:
            raise ValueError(f"Unsupported provider: {self.settings.provider}")

    async def _complete_anthropic(self, prompt: str, system: str, model: str, tools: bool) -> str:
        kwargs = {
            "model": model,
            "max_tokens": 4096,
            "system": system or "You are a helpful coding assistant.",
            "messages": [{"role": "user", "content": prompt}],
        }
        if tools:
            kwargs["tools"] = self._get_tools()
        
        response = await self.anthropic.messages.create(**kwargs)
        
        if tools and response.stop_reason == "tool_use":
            # Handle tool calls
            tool_results = []
            for content in response.content:
                if content.type == "tool_use":
                    result = await self._execute_tool(content.name, content.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": result
                    })
            
            # Continue conversation with tool results
            kwargs["messages"].append({"role": "assistant", "content": response.content})
            kwargs["messages"].append({"role": "user", "content": tool_results})
            response = await self.anthropic.messages.create(**kwargs)
        
        return response.content[0].text

    async def _complete_openai(self, prompt: str, system: str, model: str, tools: bool) -> str:
        messages = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        
        kwargs = {
            "model": model,
            "messages": messages,
        }
        if tools:
            # Convert to OpenAI function calling format
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                }
                for tool in self._get_tools()
            ]
            kwargs["tool_choice"] = "auto"
        
        response = await self.openai.chat.completions.create(**kwargs)
        
        if tools and response.choices[0].message.tool_calls:
            # Handle tool calls
            messages.append(response.choices[0].message)
            for tool_call in response.choices[0].message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = await self._execute_tool(tool_call.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            response = await self.openai.chat.completions.create(
                model=model,
                messages=messages
            )
        
        return response.choices[0].message.content or ""

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                return await self.read_file(arguments["path"])
            elif name == "write_file":
                await self.write_file(arguments["path"], arguments["content"])
                return f"Successfully wrote to {arguments['path']}"
            elif name == "list_files":
                files = self.list_files(arguments["directory"])
                return json.dumps(files)
            elif name == "run_command":
                import asyncio
                cmd = arguments["command"]
                cwd = arguments.get("cwd", str(self.settings.workspace))
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                return stdout.decode() + stderr.decode()
            else:
                return f"Unknown tool: {name}"
        except OSError as e:
            return f"Error executing {name}: {e}"
        except RuntimeError as e:
            return f"Error executing {name}: {e}"

    async def run_agent(self, prompt: str, system: str = "", model: str | None = None, max_rounds: int = 5) -> str:
        """Run an agent loop with tool calling for multiple rounds."""
        model = model or self.settings.model
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system or "You are a helpful coding assistant."},
            {"role": "user", "content": prompt}
        ]
        
        for round_num in range(max_rounds):
            if self.settings.provider == Provider.ANTHROPIC:
                response = await self.anthropic.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=messages,
                    tools=self._get_tools()
                )
                
                # Add assistant response to messages
                messages.append({"role": "assistant", "content": response.content})
                
                if response.stop_reason != "tool_use":
                    # No more tool calls, return the text content
                    for content in response.content:
                        if content.type == "text":
                            return content.text
                    return ""
                
                # Execute tool calls
                tool_results = []
                for content in response.content:
                    if content.type == "tool_use":
                        result = await self._execute_tool(content.name, content.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result
                        })
                
                messages.append({"role": "user", "content": tool_results})
                
            else:  # OpenAI
                response = await self.openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["input_schema"]
                            }
                        }
                        for tool in self._get_tools()
                    ],
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                messages.append(message.model_dump())
                
                if not message.tool_calls:
                    return message.content or ""
                
                # Execute tool calls
                for tool_call in message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    result = await self._execute_tool(tool_call.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
        
        return "Max rounds reached without completion"

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
