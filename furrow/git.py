"""Git integration for Furrow.

Furrow works in a disposable workspace, so a change is only durable once it is
committed **and** pushed to ``origin``.  This module provides safe, async git
operations that follow the version-control rules from :doc:`FURROW.md`:

* Work on the branch the session was given; never force-push or rebase a
  published branch.
* Commit at the end of every cycle that produced changes.
* Push the branch to ``origin`` before the session ends and set it to track the
  remote branch.
* Fast-forward a base branch to the run's tip and push it too.

All commands run in the workspace directory and time out like any other worker
command.  None of these functions raise: failures are reported through the
returned :class:`GitResult` so a git problem never crashes a run.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Programs that git invokes directly (e.g. through ``GIT_SSH_COMMAND``) that
#: are never allowed to run.
DENIED_GIT_HELPERS = frozenset(
    {
        "mkfs",
        "dd",
        "fdisk",
        "parted",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "init",
    },
)

#: Maximum time (seconds) for any single git command.
DEFAULT_COMMAND_TIMEOUT = 300.0


@dataclass
class GitResult:
    """Outcome of a single git command."""

    ok: bool = False
    stdout: str = ""
    stderr: str = ""
    command: str = ""
    error: str | None = None

    @property
    def combined(self) -> str:
        """Stdout plus a stderr marker, mirroring the worker's command output."""
        combined = self.stdout
        if self.stderr.strip():
            combined += "\n[stderr]\n" + self.stderr
        return combined


def _is_force_push(args: list[str]) -> bool:
    """Return True when *args* looks like a git force push."""
    if len(args) < 2 or args[0] != "git" or args[1] != "push":
        return False
    joined = " ".join(args)
    if "--force" in joined or "--force-with-lease" in joined:
        return True
    return any(a == "-f" for a in args[2:])


def _is_destructive_helper(args: list[str]) -> bool:
    """Return True when *args* would run a destructive program as a git helper."""
    for a in args:
        base = os.path.basename(a)
        if base in DENIED_GIT_HELPERS or base.startswith("mkfs."):
            return True
    return False


def _is_safe(args: list[str]) -> bool:
    """Return True when *args* pass the git safety policy."""
    return not _is_force_push(args) and not _is_destructive_helper(args)


def _format_command(args: list[str]) -> str:
    """Join args into a single shell command string."""
    return " ".join(args)


class Git:
    """Safe async git client scoped to a workspace directory."""

    def __init__(self, workspace: str | Path, timeout: float = DEFAULT_COMMAND_TIMEOUT) -> None:
        self.workspace = Path(workspace)
        self.timeout = float(timeout)

    async def _run(self, args: list[str]) -> GitResult:
        """Run one git command and return its result without ever raising."""
        if not _is_safe(args):
            return GitResult(
                ok=False,
                command=_format_command(args),
                error="command denied by git safety policy",
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            return GitResult(
                ok=False,
                command=_format_command(args),
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except (OSError, ProcessLookupError):
                pass
            return GitResult(
                ok=False,
                command=_format_command(args),
                error=f"command timed out after {self.timeout:g}s",
            )
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        ok = proc.returncode == 0
        return GitResult(
            ok=ok,
            stdout=out,
            stderr=err,
            command=_format_command(args),
            error=None if ok else f"exit {proc.returncode}",
        )

    # -- status --------------------------------------------------------------

    async def status(self) -> GitResult:
        """``git status --porcelain``."""
        return await self._run(["git", "status", "--porcelain"])

    async def is_dirty(self) -> bool:
        """True when the working tree has unstaged or staged changes."""
        result = await self.status()
        return bool(result.stdout.strip())

    async def current_branch(self) -> str:
        """The name of the current branch, or an empty string when unknown."""
        result = await self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip() if result.ok else ""

    async def remote_url(self) -> str | None:
        """The URL of ``origin``, or None when there is no remote."""
        result = await self._run(["git", "remote", "get-url", "origin"])
        return result.stdout.strip() or None if result.ok else None

    async def has_remote(self) -> bool:
        """True when the repo has an ``origin`` remote configured."""
        return (await self.remote_url()) is not None

    async def is_repo(self) -> bool:
        """True when the workspace is inside a git repository."""
        result = await self._run(["git", "rev-parse", "--is-inside-work-tree"])
        return result.ok and result.stdout.strip() == "true"

    # -- mutations -----------------------------------------------------------

    async def add(self, paths: list[str] | None = None) -> GitResult:
        """Stage *paths* (or everything when *paths* is None)."""
        args = ["git", "add"]
        if paths:
            args.extend(paths)
        else:
            args.append("-A")
        return await self._run(args)

    async def commit(self, message: str) -> GitResult:
        """Create a commit with *message*.

        Stages all changes first (``git add -A``), then commits.  Returns a
        failed result when there is nothing to commit.
        """
        staged = await self.add()
        if not staged.ok:
            return staged
        result = await self._run(["git", "commit", "-m", message])
        # ``git commit`` exits non-zero when there is nothing to commit; that
        # is not an error for our purposes.
        if not result.ok and "nothing to commit" in (result.stderr + result.stdout):
            return GitResult(ok=True, command="git commit", stdout=result.stdout)
        return result

    async def push(self, branch: str | None = None, set_upstream: bool = True) -> GitResult:
        """Push *branch* (or the current branch) to ``origin``.

        Never uses ``--force``.  When *set_upstream* is true, uses ``-u`` so
        the branch tracks its remote counterpart.
        """
        if branch is None:
            branch = await self.current_branch()
        if not branch:
            return GitResult(
                ok=False,
                command="git push",
                error="could not determine current branch",
            )
        args = ["git", "push", "origin", branch]
        if set_upstream:
            args.insert(2, "-u")
        return await self._run(args)

    async def fast_forward_base(self, base_branch: str) -> GitResult:
        """Fast-forward *base_branch* to the current tip and push it.

        A fast-forward is safe because the old tip stays in history.  This is
        used so the next run starts from the improved code rather than
        re-deriving it.
        """
        checkout = await self._run(["git", "checkout", base_branch])
        if not checkout.ok:
            return checkout
        merge = await self._run(["git", "merge", "--ff-only", "@{u}"])
        if not merge.ok:
            return merge
        return await self.push(branch=base_branch, set_upstream=False)

    # -- convenience ---------------------------------------------------------

    async def commit_and_push(self, message: str) -> GitResult:
        """Commit all changes and push the current branch to ``origin``.

        Returns the push result (or the commit result when the commit failed).
        """
        commit_result = await self.commit(message)
        if not commit_result.ok:
            return commit_result
        return await self.push()

    async def ensure_pushed(self, branch: str | None = None) -> GitResult:
        """Make sure *branch* (or the current branch) is pushed to ``origin``.

        Returns success when the branch has a remote tracking ref or when the
        push succeeds.
        """
        if branch is None:
            branch = await self.current_branch()
        if not branch:
            return GitResult(
                ok=False,
                command="git ensure-pushed",
                error="could not determine current branch",
            )
        # Check whether the remote tracking ref exists.
        tracking = await self._run(
            ["git", "rev-parse", "--verify", f"origin/{branch}"]
        )
        if tracking.ok:
            return GitResult(ok=True, command="git ensure-pushed", stdout="")
        return await self.push(branch=branch, set_upstream=True)


def git_result_from_exception(exc: BaseException, command: str = "") -> GitResult:
    """Build a :class:`GitResult` from an exception, never raising."""
    return GitResult(
        ok=False,
        command=command,
        error=f"{type(exc).__name__}: {exc}",
    )
