from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


class GitManager:
    """Manages git operations for tracking Furrow changes."""

    def __init__(self, repo_path: str | Path | None = None) -> None:
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self._git_available: bool | None = None

    async def is_available(self) -> bool:
        """Check if git is available and we're in a repo."""
        if self._git_available is not None:
            return self._git_available

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "--git-dir",
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._git_available = proc.returncode == 0
        except FileNotFoundError:
            self._git_available = False

        return self._git_available

    async def ensure_repo(self) -> bool:
        """Initialize a git repo if one doesn't exist."""
        if await self.is_available():
            return True

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "init",
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._git_available = proc.returncode == 0
            if self._git_available:
                console.print("[dim]Initialized git repository[/dim]")
            return self._git_available
        except FileNotFoundError:
            console.print("[dim yellow]Git not available - changes won't be tracked[/dim yellow]")
            return False

    async def commit_changes(self, message: str, cycle: int | None = None) -> bool:
        """Stage and commit all changes."""
        if not await self.is_available():
            return False

        try:
            # Stage all changes
            proc = await asyncio.create_subprocess_exec(
                "git", "add", "-A",
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            # Create commit message
            commit_msg = f"[Furrow] {message}"
            if cycle:
                commit_msg = f"[Furrow Cycle {cycle}] {message}"

            # Commit
            proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", commit_msg,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                console.print(f"[dim]Committed: {commit_msg}[/dim]")
                return True
            else:
                # Nothing to commit is OK
                if "nothing to commit" in stderr.decode().lower():
                    return True
                console.print(f"[dim yellow]Git commit failed: {stderr.decode()}[/dim yellow]")
                return False

        except FileNotFoundError:
            return False

    async def get_status(self) -> dict[str, list[str]]:
        """Get current git status."""
        if not await self.is_available():
            return {"modified": [], "untracked": [], "staged": []}

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "status", "--porcelain",
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            status = {"modified": [], "untracked": [], "staged": []}
            for line in stdout.decode().strip().split("\n"):
                if not line:
                    continue
                status_code = line[:2]
                filename = line[3:]

                if status_code == "??":
                    status["untracked"].append(filename)
                elif status_code.startswith("M"):
                    status["staged"].append(filename)
                elif status_code.endswith("M"):
                    status["modified"].append(filename)

            return status
        except FileNotFoundError:
            return {"modified": [], "untracked": [], "staged": []}

    async def create_checkpoint(self, label: str) -> bool:
        """Create a tagged checkpoint for easy rollback."""
        if not await self.is_available():
            return False

        try:
            tag_name = f"furrow-checkpoint-{label}"
            proc = await asyncio.create_subprocess_exec(
                "git", "tag", "-a", tag_name, "-m", f"Furrow checkpoint: {label}",
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False
