"""Git integration for Furrow.

This module provides Git operations for the worker agent: commit, diff,
branch, status, and log. These operations are safe to run in the workspace
and help track changes made by the agent.
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import Any


async def _run_git(workspace: Path, *args: str) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    cmd = ["git", *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(workspace),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    except asyncio.TimeoutError:
        return -1, "", "git command timed out"
    except (OSError, ValueError) as exc:
        return -1, "", f"git error: {exc}"


async def git_status(workspace: Path) -> str:
    """Get the git status of the workspace."""
    returncode, stdout, stderr = await _run_git(workspace, "status", "--porcelain")
    if returncode != 0:
        return f"error: {stderr}"
    return stdout or "(clean)"


async def git_diff(workspace: Path, staged: bool = False) -> str:
    """Get the git diff of the workspace."""
    args = ["diff"]
    if staged:
        args.append("--cached")
    returncode, stdout, stderr = await _run_git(workspace, *args)
    if returncode != 0:
        return f"error: {stderr}"
    return stdout or "(no changes)"


async def git_log(workspace: Path, max_count: int = 10) -> str:
    """Get the git log of the workspace."""
    returncode, stdout, stderr = await _run_git(
        workspace, "log", f"-{max_count}", "--oneline", "--decorate"
    )
    if returncode != 0:
        return f"error: {stderr}"
    return stdout or "(no commits)"


async def git_branch(workspace: Path) -> str:
    """Get the current branch name."""
    returncode, stdout, stderr = await _run_git(
        workspace, "branch", "--show-current"
    )
    if returncode != 0:
        return f"error: {stderr}"
    return stdout.strip() or "(detached)"


async def git_commit(workspace: Path, message: str, add_all: bool = True) -> str:
    """Commit changes in the workspace."""
    if add_all:
        returncode, stdout, stderr = await _run_git(workspace, "add", "-A")
        if returncode != 0:
            return f"error staging files: {stderr}"

    returncode, stdout, stderr = await _run_git(
        workspace, "commit", "-m", message
    )
    if returncode != 0:
        return f"error committing: {stderr}"
    return stdout or "committed"


async def git_checkout(workspace: Path, branch: str, create: bool = False) -> str:
    """Checkout or create a branch."""
    args = ["checkout"]
    if create:
        args.append("-b")
    args.append(branch)
    returncode, stdout, stderr = await _run_git(workspace, *args)
    if returncode != 0:
        return f"error: {stderr}"
    return stdout or f"switched to {branch}"


async def git_is_repo(workspace: Path) -> bool:
    """Check if the workspace is a git repository."""
    returncode, _, _ = await _run_git(workspace, "rev-parse", "--is-inside-work-tree")
    return returncode == 0


# Tool definitions for the worker
GIT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "git_status",
        "description": "Show the working tree status (modified, staged, untracked files).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "git_diff",
        "description": "Show changes between commits, commit and working tree, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "staged": {"type": "boolean", "description": "Show staged changes only"},
            },
            "required": [],
        },
    },
    {
        "name": "git_log",
        "description": "Show commit logs.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_count": {"type": "integer", "description": "Maximum number of commits to show"},
            },
            "required": [],
        },
    },
    {
        "name": "git_branch",
        "description": "Show the current branch name.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "git_commit",
        "description": "Commit staged changes with a message.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message"},
                "add_all": {"type": "boolean", "description": "Stage all changes before committing"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "git_checkout",
        "description": "Switch branches or create a new branch.",
        "parameters": {
            "type": "object",
            "properties": {
                "branch": {"type": "string", "description": "Branch name"},
                "create": {"type": "boolean", "description": "Create new branch"},
            },
            "required": ["branch"],
        },
    },
]


async def execute_git_tool(workspace: Path, name: str, args: dict[str, Any]) -> str:
    """Execute a git tool and return the result."""
    if name == "git_status":
        return await git_status(workspace)
    if name == "git_diff":
        return await git_diff(workspace, args.get("staged", False))
    if name == "git_log":
        return await git_log(workspace, args.get("max_count", 10))
    if name == "git_branch":
        return await git_branch(workspace)
    if name == "git_commit":
        return await git_commit(workspace, args.get("message", ""), args.get("add_all", True))
    if name == "git_checkout":
        return await git_checkout(workspace, args.get("branch", ""), args.get("create", False))
    return f"unknown git tool: {name}"