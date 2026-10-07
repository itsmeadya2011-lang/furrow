from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional


class GitRepo:
    def __init__(self, path: Path | str = ".") -> None:
        self.path = Path(path).resolve()

    async def _run(self, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=self.path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode or 0, stdout.decode(), stderr.decode()

    async def is_repo(self) -> bool:
        code, _, _ = await self._run("rev-parse", "--is-inside-work-tree")
        return code == 0

    async def init(self) -> bool:
        code, _, _ = await self._run("init")
        return code == 0

    async def status(self) -> str:
        _, stdout, _ = await self._run("status", "--short")
        return stdout

    async def add(self, paths: list[str] | str = ".") -> bool:
        if isinstance(paths, str):
            paths = [paths]
        code, _, _ = await self._run("add", *paths)
        return code == 0

    async def commit(self, message: str) -> bool:
        code, _, _ = await self._run("commit", "-m", message)
        return code == 0

    async def diff(self, staged: bool = False) -> str:
        args = ["diff"]
        if staged:
            args.append("--staged")
        _, stdout, _ = await self._run(*args)
        return stdout

    async def log(self, limit: int = 10) -> str:
        _, stdout, _ = await self._run("log", "--oneline", f"-{limit}")
        return stdout

    async def branch(self) -> str:
        _, stdout, _ = await self._run("branch", "--show-current")
        return stdout.strip()

    async def has_changes(self) -> bool:
        _, stdout, _ = await self._run("status", "--porcelain")
        return bool(stdout.strip())

    async def commit_all(self, message: str) -> bool:
        if not await self.has_changes():
            return False
        await self.add(".")
        return await self.commit(message)