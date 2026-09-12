from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any

import aiofiles
import httpx
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
        content = response.content
        if not content or not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif hasattr(block, "text") and isinstance(getattr(block, "text", None), str):
                parts.append(block.text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)

    async def _complete_ollama(self, prompt: str, system: str, model: str) -> str:
        url = self.settings.ollama_base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": [],
            "stream": False,
        }
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": prompt})
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(url, json=payload, timeout=60.0)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ollama request failed: {e.response.status_code} {e.response.text}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"Ollama connection failed: {e}") from e
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Ollama returned invalid JSON: {e}") from e
        if "response" in data:
            return data["response"]
        if "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content", "")
        return ""

    async def _complete_openai(self, prompt: str, system: str, model: str) -> str:
        response = await self.openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful coding assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        return content or ""

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path, "r") as f:
            return await f.read()

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)

    async def run_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        if not command or not command.strip():
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": "Empty command provided",
            }
        cmd_cwd = Path(cwd) if cwd is not None else self.settings.workspace
        args = shlex.split(command)
        if not args:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": "Command parsed to empty argument list",
            }
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(cmd_cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout}s: {command}",
                }
            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
            }
        except FileNotFoundError:
            return {
                "returncode": 127,
                "stdout": "",
                "stderr": f"Command not found: {args[0]}",
            }
        except Exception as e:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": str(e),
            }

    def list_files(self, directory: str | Path) -> list[str]:
        p = Path(directory)
        if not p.exists():
            return []
        ignored_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "env",
            ".env",
            "dist",
            "build",
            ".mypy_cache",
            ".pytest_cache",
            ".next",
            ".nuxt",
            "target",
            "bin",
            "obj",
            ".hg",
            ".svn",
            ".tox",
        }
        results = []
        for root, dirs, files in os.walk(p):
            dirs[:] = [
                d
                for d in dirs
                if d not in ignored_dirs and not (d.startswith(".") and d != ".")
            ]
            for f in files:
                results.append(str(Path(root, f).relative_to(p)))
        return results
