"""Git integration for Furrow.

Provides utilities to commit and push changes after each cycle, with
configurable behavior and safe defaults.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
from pathlib import Path
from typing import Any

from furrow.config import Settings
from furrow.logging import get_logger

logger = get_logger(__name__)


class GitError(Exception):
    """Raised when a Git operation fails."""
    pass


async def run_git(
    args: list[str],
    cwd: Path,
    timeout: float = 30.0,
) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise GitError(f"git {' '.join(args)} timed out after {timeout}s") from exc
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def run_git_sync(
    args: list[str],
    cwd: Path,
    timeout: float = 30.0,
) -> tuple[int, str, str]:
    """Run a git command synchronously and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args)} timed out after {timeout}s") from exc
    except OSError as exc:
        raise GitError(f"failed to run git: {exc}") from exc


async def is_git_repo(cwd: Path) -> bool:
    """Check if the directory is a Git repository."""
    code, _, _ = await run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    return code == 0


async def has_remote(cwd: Path, remote: str = "origin") -> bool:
    """Check if the repository has a configured remote."""
    code, stdout, _ = await run_git(["remote"], cwd)
    if code != 0:
        return False
    return remote in stdout.split()


async def get_current_branch(cwd: Path) -> str | None:
    """Get the name of the current branch."""
    code, stdout, _ = await run_git(["branch", "--show-current"], cwd)
    if code != 0:
        return None
    branch = stdout.strip()
    return branch if branch else None


async def get_status(cwd: Path) -> dict[str, list[str]]:
    """Get the git status: modified, untracked, staged files."""
    code, stdout, _ = await run_git(["status", "--porcelain"], cwd)
    if code != 0:
        return {"modified": [], "untracked": [], "staged": []}

    modified = []
    untracked = []
    staged = []

    for line in stdout.splitlines():
        if len(line) < 3:
            continue
        status = line[:2]
        path = line[3:].strip()
        if status[0] != " ":
            staged.append(path)
        if status[1] != " ":
            if status[1] == "?":
                untracked.append(path)
            else:
                modified.append(path)

    return {"modified": modified, "untracked": untracked, "staged": staged}


async def has_changes(cwd: Path) -> bool:
    """Check if there are any uncommitted changes."""
    status = await get_status(cwd)
    return bool(status["modified"] or status["untracked"] or status["staged"])


async def diff(cwd: Path, staged: bool = False) -> str:
    """Get the diff of changes."""
    args = ["diff"]
    if staged:
        args.append("--cached")
    code, stdout, stderr = await run_git(args, cwd)
    if code != 0:
        raise GitError(f"git diff failed: {stderr}")
    return stdout


async def add_all(cwd: Path) -> None:
    """Stage all changes (modified, deleted, untracked)."""
    code, _, stderr = await run_git(["add", "-A"], cwd)
    if code != 0:
        raise GitError(f"git add failed: {stderr}")


async def commit(
    cwd: Path,
    message: str,
    author: str | None = None,
) -> str:
    """Create a commit with the given message. Returns the commit hash."""
    args = ["commit", "-m", message]
    if author:
        args.extend(["--author", author])
    code, stdout, stderr = await run_git(args, cwd)
    if code != 0:
        # Check if it's just "nothing to commit"
        if "nothing to commit" in stderr.lower():
            return ""
        raise GitError(f"git commit failed: {stderr}")

    # Extract commit hash from output
    for line in stdout.splitlines():
        if line.startswith("["):
            # Format: [branch hash] message
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].rstrip("]")
    return ""


async def push(
    cwd: Path,
    remote: str = "origin",
    branch: str | None = None,
    force: bool = False,
) -> None:
    """Push the current branch to the remote."""
    args = ["push"]
    if force:
        args.append("--force")
    if remote:
        args.append(remote)
    if branch:
        args.append(branch)
    code, _, stderr = await run_git(args, cwd)
    if code != 0:
        raise GitError(f"git push failed: {stderr}")


async def fetch(
    cwd: Path,
    remote: str = "origin",
) -> None:
    """Fetch from the remote."""
    code, _, stderr = await run_git(["fetch", remote], cwd)
    if code != 0:
        raise GitError(f"git fetch failed: {stderr}")


async def fast_forward(
    cwd: Path,
    remote: str = "origin",
    branch: str | None = None,
) -> bool:
    """Fast-forward the current branch to the remote. Returns True if fast-forwarded."""
    current = await get_current_branch(cwd)
    if not current:
        return False

    target_branch = branch or current
    code, _, _ = await run_git(["merge", "--ff-only", f"{remote}/{target_branch}"], cwd)
    return code == 0


async def get_commit_message_for_cycle(
    cycle: int,
    goal: str,
    tasks: list[dict[str, Any]],
    files_changed: list[str],
) -> str:
    """Generate a commit message for a cycle."""
    # Truncate goal for subject line
    subject_goal = goal.strip().split("\n")[0]
    if len(subject_goal) > 60:
        subject_goal = subject_goal[:57] + "..."

    lines = [
        f"Cycle {cycle}: {subject_goal}",
        "",
        f"Goal: {goal.strip()}",
        "",
        "Tasks completed:",
    ]
    for task in tasks:
        status = task.get("status", "unknown")
        if status == "completed":
            lines.append(f"  - {task.get('id')}: {task.get('description', '')}")
    if files_changed:
        lines.append("")
        lines.append("Files changed:")
        for f in sorted(files_changed):
            lines.append(f"  - {f}")
    return "\n".join(lines)


class GitManager:
    """High-level Git operations for Furrow runs."""

    def __init__(
        self,
        workspace: Path,
        settings: Settings | None = None,
        auto_commit: bool = True,
        auto_push: bool = True,
        remote: str = "origin",
        author: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.settings = settings
        self.auto_commit = auto_commit
        self.auto_push = auto_push
        self.remote = remote
        self.author = author

    async def maybe_commit_and_push(
        self,
        cycle: int,
        goal: str,
        tasks: list[dict[str, Any]],
        files_changed: list[str],
    ) -> dict[str, Any]:
        """Commit and push changes if there are any and auto-commit/push are enabled.

        Returns a dict with the results of the operations.
        """
        result: dict[str, Any] = {
            "committed": False,
            "pushed": False,
            "commit_hash": None,
            "error": None,
        }

        if not self.auto_commit:
            return result

        # Check if we're in a git repo
        if not await is_git_repo(self.workspace):
            logger.debug("Not a git repository, skipping commit/push")
            return result

        # Check if there are changes
        if not await has_changes(self.workspace):
            logger.debug("No changes to commit")
            return result

        try:
            # Stage all changes
            await add_all(self.workspace)

            # Generate commit message
            message = await get_commit_message_for_cycle(
                cycle, goal, tasks, files_changed
            )

            # Commit
            commit_hash = await commit(self.workspace, message, self.author)
            if commit_hash:
                result["committed"] = True
                result["commit_hash"] = commit_hash
                logger.info("Committed cycle %d changes: %s", cycle, commit_hash[:8])

                # Push if enabled
                if self.auto_push:
                    if await has_remote(self.workspace, self.remote):
                        await push(self.workspace, self.remote)
                        result["pushed"] = True
                        logger.info("Pushed to %s", self.remote)
                    else:
                        logger.debug("No remote '%s' configured, skipping push", self.remote)
            else:
                logger.debug("Nothing to commit")

        except GitError as exc:
            result["error"] = str(exc)
            logger.warning("Git operation failed: %s", exc)

        return result

    async def fast_forward_base_branch(self, base_branch: str) -> bool:
        """Fast-forward the base branch to the current tip.

        This is useful when the session was given a base branch and we want
        to update it so the next run starts from the improved code.
        """
        if not await is_git_repo(self.workspace):
            return False

        current = await get_current_branch(self.workspace)
        if not current:
            return False

        try:
            await fetch(self.workspace, self.remote)
            # Switch to base branch
            code, _, stderr = await run_git(
                ["checkout", base_branch], self.workspace
            )
            if code != 0:
                logger.warning("Could not checkout base branch %s: %s", base_branch, stderr)
                return False

            # Fast-forward
            ff_code, _, _ = await run_git(
                ["merge", "--ff-only", f"{self.remote}/{current}"], self.workspace
            )
            if ff_code == 0:
                logger.info("Fast-forwarded %s to %s", base_branch, current)
                return True
            else:
                logger.warning("Could not fast-forward %s to %s", base_branch, current)
                return False
        except GitError as exc:
            logger.warning("Fast-forward failed: %s", exc)
            return False
        finally:
            # Return to original branch
            with contextlib.suppress(Exception):
                await run_git(["checkout", current], self.workspace)


def create_git_manager(
    workspace: Path,
    settings: Settings | None = None,
    auto_commit: bool = True,
    auto_push: bool = True,
) -> GitManager:
    """Create a GitManager with settings from environment or defaults.

    Reads FURROW_GIT_AUTO_COMMIT, FURROW_GIT_AUTO_PUSH, FURROW_GIT_REMOTE,
    FURROW_GIT_AUTHOR from environment if not explicitly provided.
    """
    if settings is not None:
        # Settings object doesn't have git settings yet, but could be extended
        pass

    auto_commit = auto_commit and os.getenv("FURROW_GIT_AUTO_COMMIT", "true").lower() != "false"
    auto_push = auto_push and os.getenv("FURROW_GIT_AUTO_PUSH", "true").lower() != "false"
    remote = os.getenv("FURROW_GIT_REMOTE", "origin")
    author = os.getenv("FURROW_GIT_AUTHOR")

    return GitManager(
        workspace=workspace,
        settings=settings,
        auto_commit=auto_commit,
        auto_push=auto_push,
        remote=remote,
        author=author,
    )