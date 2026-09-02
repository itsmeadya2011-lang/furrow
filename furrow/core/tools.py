from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}
        self.register("read_file", self._read_file)
        self.register("write_file", self._write_file)
        self.register("list_files", self._list_files)
        self.register("run_command", self._run_command)

    def register(self, name: str, func: Callable[..., str]) -> None:
        self._tools[name] = func

    async def execute(self, name: str, *args: str) -> str:
        func = self._tools.get(name)
        if func is None:
            return f"Error: unknown tool {name}"
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args)
            return func(*args)
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    async def _read_file(path: str) -> str:
        import aiofiles

        async with aiofiles.open(path, "r") as f:
            return await f.read()

    @staticmethod
    async def _write_file(path: str, content: str) -> str:
        import aiofiles

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(p, "w") as f:
            await f.write(content)
        return f"Wrote {len(content)} bytes to {path}"

    @staticmethod
    def _list_files(directory: str) -> str:
        p = Path(directory)
        if not p.exists():
            return f"Error: directory {directory} does not exist"
        files = [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
        return "\n".join(files) if files else "(empty directory)"

    @staticmethod
    async def _run_command(command: str) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                return stdout.decode("utf-8", errors="replace")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return "Error: command timed out after 30 seconds"
        except Exception as e:
            return f"Error: {e}"
