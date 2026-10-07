from __future__ import annotations

import asyncio
from pathlib import Path


async def run_git(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    cmd = ["git"] + args
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


async def is_git_repo(path: Path) -> bool:
    """Check if a path is a git repository."""
    code, _, _ = await run_git(["rev-parse", "--git-dir"], cwd=path)
    return code == 0


async def get_changed_files(path: Path) -> list[str]:
    """Get list of changed files in the working directory."""
    code, stdout, _ = await run_git(["status", "--porcelain"], cwd=path)
    if code != 0:
        return []
    files = []
    for line in stdout.strip().split("\n"):
        if line:
            # Format: "XY filename" where XY are status codes
            files.append(line[3:].strip())
    return files


async def get_diff(path: Path) -> str:
    """Get the diff of staged and unstaged changes."""
    code, stdout, _ = await run_git(["diff", "HEAD"], cwd=path)
    if code != 0:
        return ""
    return stdout


async def stage_all(path: Path) -> bool:
    """Stage all changes."""
    code, _, _ = await run_git(["add", "-A"], cwd=path)
    return code == 0


async def commit(path: Path, message: str) -> bool:
    """Commit staged changes with a message."""
    code, _, _ = await run_git(["commit", "-m", message], cwd=path)
    return code == 0


async def auto_commit(path: Path, message: str) -> bool:
    """Automatically stage and commit all changes."""
    if not await is_git_repo(path):
        return False
    changed = await get_changed_files(path)
    if not changed:
        return False
    if not await stage_all(path):
        return False
    return await commit(path, message)


async def get_recent_commits(path: Path, count: int = 5) -> list[str]:
    """Get recent commit messages."""
    code, stdout, _ = await run_git(["log", f"--oneline", f"-{count}"], cwd=path)
    if code != 0:
        return []
    return stdout.strip().split("\n")