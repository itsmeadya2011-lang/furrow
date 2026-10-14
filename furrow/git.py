from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class GitManager:
    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace or Path.cwd()
        self._initial_commit: str | None = None

    async def _run(self, *args: str) -> tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)."""
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout.decode(), stderr.decode()

    async def is_repo(self) -> bool:
        """Check if the workspace is a git repository."""
        code, _, _ = await self._run("rev-parse", "--git-dir")
        return code == 0

    async def init(self) -> bool:
        """Initialize a git repository if not already initialized."""
        if await self.is_repo():
            return True
        code, _, stderr = await self._run("init")
        if code != 0:
            logger.error("Failed to init git repo", error=stderr)
            return False
        await self._run("config", "user.email", "furrow@local")
        await self._run("config", "user.name", "Furrow Agent")
        return True

    async def get_status(self) -> str:
        """Get git status."""
        code, stdout, _ = await self._run("status", "--porcelain")
        return stdout if code == 0 else ""

    async def get_diff(self, staged: bool = False) -> str:
        """Get git diff."""
        args = ["diff"]
        if staged:
            args.append("--staged")
        code, stdout, _ = await self._run(*args)
        return stdout if code == 0 else ""

    async def add_all(self) -> bool:
        """Stage all changes."""
        code, _, stderr = await self._run("add", "-A")
        if code != 0:
            logger.error("Failed to add files", error=stderr)
            return False
        return True

    async def commit(self, message: str) -> str | None:
        """Commit staged changes and return commit hash."""
        code, stdout, stderr = await self._run("commit", "-m", message)
        if code != 0:
            logger.error("Failed to commit", error=stderr)
            return None
        # Extract commit hash
        code, stdout, _ = await self._run("rev-parse", "HEAD")
        return stdout.strip() if code == 0 else None

    async def auto_commit(self, message: str) -> str | None:
        """Stage all changes and commit."""
        if not await self.add_all():
            return None
        status = await self.get_status()
        if not status.strip():
            logger.info("No changes to commit")
            return None
        return await self.commit(message)

    async def save_checkpoint(self, cycle: int, goal: str) -> str | None:
        """Create a checkpoint commit."""
        message = f"furrow: checkpoint cycle {cycle} - {goal[:50]}"
        return await self.auto_commit(message)

    async def rollback_to_checkpoint(self, commit_hash: str) -> bool:
        """Rollback to a specific commit."""
        code, _, stderr = await self._run("reset", "--hard", commit_hash)
        if code != 0:
            logger.error("Failed to rollback", error=stderr)
            return False
        return True

    async def get_initial_commit(self) -> str | None:
        """Get the initial commit hash before Furrow started."""
        if self._initial_commit:
            return self._initial_commit
        code, stdout, _ = await self._run("rev-parse", "HEAD")
        if code == 0:
            self._initial_commit = stdout.strip()
        return self._initial_commit

    async def list_checkpoints(self) -> list[dict]:
        """List all Furrow checkpoint commits."""
        code, stdout, _ = await self._run("log", "--oneline", "--grep=furrow: checkpoint")
        if code != 0:
            return []
        checkpoints = []
        for line in stdout.strip().split("\n"):
            if line:
                parts = line.split(" ", 1)
                checkpoints.append(
                    {"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""}
                )
        return checkpoints
