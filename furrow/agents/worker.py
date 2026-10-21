from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import signal
from pathlib import Path
from typing import Any

from furrow.agents.prompts import WORKER_SYSTEM, WORKER_TOOLS
from furrow.config import Settings, TaskModel, WorkerResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient, LLMResponse, ToolCall

# Substring patterns that are denied in a command. They are matched
# case-insensitively against the normalized command.
DENIED_COMMANDS = (
    "rm -rf /",
    "rm -fr /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf $home",
    ":(){:|:&};:",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "chmod -R 777 /",
    "> /dev/sda",
)

# Programs that are denied by name: the basename of the first token of a
# segment. ``mkfs`` variants (``mkfs.ext4`` and friends) are denied as well.
DENIED_PROGRAMS = frozenset(
    {
        "dd",
        "fdisk",
        "halt",
        "init",
        "mkfs",
        "parted",
        "poweroff",
        "reboot",
        "shutdown",
    }
)

# Maximum number of entries returned by the ``list_files`` tool.
MAX_LIST_ENTRIES = 500

# Shell separators a command is split on.
_SEGMENT_SPLIT_RE = re.compile(r"(?:&&|\|\||;|\||\n|\r)+")


def _normalize_command(command: str) -> str:
    """Normalize command for matching: lowercase, collapse whitespace,
    strip surrounding whitespace/quotes."""
    if not command:
        return ""
    cmd = command.strip()
    # Strip surrounding quotes (single/double)
    while len(cmd) >= 2 and ((cmd[0] == cmd[-1]) and cmd[0] in ('"', "'")):
        cmd = cmd[1:-1].strip()
    cmd = cmd.lower()
    # Collapse runs of whitespace to single space
    cmd = re.sub(r'\s+', ' ', cmd)
    return cmd


def _split_command_segments(command: str) -> list[str]:
    """Split command into segments on shell separators ;, &&, ||, |, and newlines."""
    if not command:
        return []
    segments = _SEGMENT_SPLIT_RE.split(command)
    return [s for s in segments if s]  # Remove empty segments


def _get_program_basename(segment: str) -> str | None:
    """Get basename of first token in segment."""
    seg = segment.strip()
    if not seg:
        return None
    try:
        parts = shlex.split(seg)
    except ValueError:
        parts = seg.split()
    if not parts:
        return None
    first = parts[0]
    return os.path.basename(first)


def _is_dangerous_segment(segment: str, workspace: Path) -> bool:
    """Check if a normalized segment is dangerous."""
    norm_seg = _normalize_command(segment)
    if not norm_seg:
        return False

    # DENY by PROGRAM NAME
    prog = _get_program_basename(segment)
    if prog:
        prog_norm = prog.lower()
        if prog_norm in DENIED_PROGRAMS or prog_norm.startswith("mkfs."):
            return True

    # DENY by PATTERN
    for denied in DENIED_COMMANDS:
        if denied.lower() in norm_seg:
            return True

    # DENY force-pushing for git push
    if "git" in norm_seg and "push" in norm_seg:
        # Covers --force, --force-with-lease and --force-with-lease=<ref>.
        if "--force" in norm_seg:
            return True
        # A short -f flag only counts as a shell token, so that ordinary
        # commands (e.g. "ls -f") are not mistaken for a force push.
        try:
            parts = shlex.split(segment)
        except ValueError:
            parts = segment.split()
        if any(part.lower() == "-f" for part in parts):
            return True

    # DENY redirection to absolute path outside workspace
    redir_pattern = r'>>?\s*([^\s;|&]+)'
    for m in re.finditer(redir_pattern, segment):
        target = m.group(1)
        target_clean = target.strip().strip('"').strip("'")
        if not target_clean:
            continue
        target_path = Path(target_clean)
        if target_path.is_absolute():
            try:
                target_path.resolve().relative_to(workspace.resolve())
            except ValueError:
                return True

    return False


def is_command_denied(command: str, workspace: Path) -> bool:
    """Check if command is denied by security policy."""
    if not command or not command.strip():
        return False
    # A denied pattern that contains a shell separator (the fork bomb) is
    # broken apart by the segment split below, so it is matched against the
    # whole command.
    normalized = _normalize_command(command)
    for denied in DENIED_COMMANDS:
        if _SEGMENT_SPLIT_RE.search(denied) and denied.lower() in normalized:
            return True
    segments = _split_command_segments(command)
    if not segments:
        segments = [command]
    return any(_is_dangerous_segment(seg, workspace) for seg in segments)


def resolve_in_workspace(workspace: Path, path: str) -> Path | None:
    """Return the absolute path for `path` inside `workspace`, or None when it escapes."""
    workspace = workspace.resolve()
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        try:
            p.relative_to(workspace)
        except ValueError:
            return None
        return p
    # reject traversal
    candidate = (workspace / p).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError:
        return None
    return candidate


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
        goal: str = "",
    ) -> None:
        self.task = task
        self.settings = settings if settings is not None else global_settings
        self.workspace = Path(workspace) if workspace is not None else Path(self.settings.workspace)
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.goal = goal
        self.files_changed: set[str] = set()

    async def run(self) -> WorkerResult:
        try:
            return await self._run()
        except Exception as exc:  # pragma: no cover - defensive
            return WorkerResult(summary=str(exc), success=False, error=str(exc), iterations=0)

    async def _run(self) -> WorkerResult:
        max_iters = self.settings.max_worker_iterations
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._initial_text(),
            }
        ]
        iterations = 0
        for i in range(max_iters):
            iterations = i + 1
            response: LLMResponse = await self.client.chat(
                messages, tools=WORKER_TOOLS, system=WORKER_SYSTEM, model=self.settings.worker_model
            )
            if not response.tool_calls:
                summary = response.text.strip() if response.text.strip() else "Task completed."
                return WorkerResult(
                    summary=summary,
                    success=True,
                    files_changed=sorted(self.files_changed),
                    iterations=iterations,
                )
            # append assistant turn
            blocks: list[dict[str, Any]] = []
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            messages.append({"role": "assistant", "content": blocks})

            # execute tools
            result_blocks: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                result = await self._execute_tool(tc)
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                )
            messages.append({"role": "user", "content": result_blocks})

        # budget exhausted: the worker was truncated, so this is NOT a success.
        # Reporting success here would make the orchestrator mark the task as
        # completed instead of retrying it.
        return WorkerResult(
            summary="worker iteration budget exhausted",
            success=False,
            files_changed=sorted(self.files_changed),
            iterations=max_iters,
            error="worker iteration budget exhausted",
        )

    def _initial_text(self) -> str:
        parts = []
        if self.goal:
            parts.append(f"Overall goal: {self.goal}")
        parts.append(f"Task: {self.task.description}")
        parts.append(f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}")
        parts.append(f"Workspace: {self.workspace}")
        return "\n".join(parts)

    async def _execute_tool(self, tc: ToolCall) -> str:
        name = tc.name
        args = tc.arguments or {}
        if name == "read_file":
            return await self._read_file(str(args.get("path", "")))
        if name == "write_file":
            return await self._write_file(str(args.get("path", "")), str(args.get("content", "")))
        if name == "list_files":
            return self._list_files(
                str(args.get("path", "")) if args.get("path") is not None else "."
            )
        if name == "run_command":
            return await self._run_command(str(args.get("command", "")))
        return f"unknown tool: {name}"

    async def _read_file(self, path: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            content = await self.client.read_file(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        if len(content) > 20000:
            content = content[:20000]
        return content

    async def _write_file(self, path: str, content: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(resolved, content)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        rel = str(resolved.relative_to(self.workspace.resolve()))
        self.files_changed.add(rel)
        return f"wrote {path} ({len(content.encode('utf-8'))} bytes)"

    def _list_files(self, path: str) -> str:
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            entries = self.client.list_files(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        # Cap the listing at MAX_LIST_ENTRIES so huge trees cannot flood the context.
        rels = []
        base = self.workspace.resolve()
        for e in entries[:MAX_LIST_ENTRIES]:
            try:
                rels.append(str(Path(e).relative_to(base)) if Path(e).is_absolute() else e)
            except ValueError:
                rels.append(str(e))
        return "\n".join(rels)

    async def _run_command(self, command: str) -> str:
        if not command.strip():
            return "error: empty command"
        # The hardened check subsumes the legacy pattern list: it normalizes
        # the command, checks every segment against DENIED_COMMANDS and
        # DENIED_PROGRAMS, and denies force pushes and redirections outside
        # the workspace.
        if is_command_denied(command, self.workspace):
            return "error: command denied"
        try:
            parts = shlex.split(command)
        except ValueError:
            return "error: could not parse command"
        if not parts:
            return "error: empty command"
        timeout = min(self.settings.request_timeout, 120.0)
        # On POSIX the command runs in its own session/process group so that a
        # timeout can kill the whole tree (children and grandchildren included)
        # instead of leaking orphans. On other platforms we fall back to
        # killing the direct child only.
        new_session = os.name == "posix"
        spawn_kwargs: dict[str, Any] = {"start_new_session": True} if new_session else {}
        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **spawn_kwargs,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await self._kill_process_tree(proc, new_session)
            return "command timed out"
        combined = stdout.decode("utf-8", errors="replace") + stderr.decode(
            "utf-8", errors="replace"
        )
        if len(combined) > 20000:
            combined = combined[:20000]
        return combined

    @staticmethod
    async def _kill_process_tree(proc: asyncio.subprocess.Process, new_session: bool) -> None:
        """Kill a timed-out command together with every process it spawned."""
        if new_session:
            try:
                pgid = os.getpgid(proc.pid)
            except (OSError, AttributeError):
                pgid = None
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, AttributeError):
                    pass
                else:
                    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    return
        # Fallback: no process group available (or killing it failed).
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
