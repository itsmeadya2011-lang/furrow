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

DENIED_COMMANDS = (
    "rm -rf /",
    "rm -rf ~",
    ":(){:|:&};:",
    "mkfs",
    "dd if=",
    "shutdown",
    "reboot",
    "chmod -R 777 /",
    "> /dev/sda",
)

# Maximum number of entries returned by the ``list_files`` tool.
MAX_LIST_ENTRIES = 500

# Maximum number of lines returned by ``grep_search``.
MAX_GREP_LINES = 200

# Maximum size of content accepted by ``write_file`` / ``edit_file``.
MAX_WRITE_SIZE = 100_000

# Maximum size of old_text/new_text for ``edit_file``.
MAX_EDIT_TEXT_SIZE = 50_000


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
                messages, tools=WORKER_TOOLS, system=WORKER_SYSTEM,
                model=self.settings.worker_model, role="worker",
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
        if name == "edit_file":
            return await self._edit_file(
                str(args.get("path", "")),
                str(args.get("old_text", "")),
                str(args.get("new_text", "")),
            )
        if name == "list_files":
            return self._list_files(
                str(args.get("path", "")) if args.get("path") is not None else "."
            )
        if name == "grep_search":
            return self._grep_search(
                str(args.get("path", "")) if args.get("path") is not None else ".",
                str(args.get("pattern", "")),
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
        if len(content.encode("utf-8")) > MAX_WRITE_SIZE:
            return f"error: content too large (max {MAX_WRITE_SIZE} bytes)"
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

    async def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace the first occurrence of *old_text* with *new_text* in *path*."""
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        if len(old_text) > MAX_EDIT_TEXT_SIZE:
            return f"error: old_text too large (max {MAX_EDIT_TEXT_SIZE} bytes)"
        if len(new_text) > MAX_EDIT_TEXT_SIZE:
            return f"error: new_text too large (max {MAX_EDIT_TEXT_SIZE} bytes)"
        try:
            content = await self.client.read_file(resolved)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        count = content.count(old_text)
        if count == 0:
            return f"error: old_text not found in {path}"
        if count > 1:
            return f"error: old_text is ambiguous ({count} matches in {path}); provide more context"
        new_content = content.replace(old_text, new_text, 1)
        if len(new_content.encode("utf-8")) > MAX_WRITE_SIZE:
            return f"error: resulting file too large (max {MAX_WRITE_SIZE} bytes)"
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(resolved, new_content)
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        rel = str(resolved.relative_to(self.workspace.resolve()))
        self.files_changed.add(rel)
        return f"edited {path} ({len(new_content.encode('utf-8'))} bytes)"

    def _grep_search(self, path: str, pattern: str) -> str:
        """Search file contents under *path* for *pattern* (regex)."""
        if not pattern:
            return "error: pattern is required"
        resolved = resolve_in_workspace(self.workspace, path)
        if resolved is None:
            return "error: path escapes workspace"
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"error: invalid regex: {exc}"
        matches: list[str] = []
        base = self.workspace.resolve()
        try:
            for entry in self.client.list_files(resolved):
                try:
                    file_path = Path(entry)
                    if not file_path.is_absolute():
                        file_path = resolved / file_path
                    if not file_path.is_file():
                        continue
                    with open(file_path, encoding="utf-8", errors="replace") as fh:
                        for line_no, line in enumerate(fh, start=1):
                            if regex.search(line):
                                try:
                                    rel = str(file_path.relative_to(base))
                                except ValueError:
                                    rel = str(file_path)
                                matches.append(f"{rel}:{line_no}:{line.rstrip()}")
                                if len(matches) >= MAX_GREP_LINES:
                                    break
                except (OSError, ValueError):
                    continue
                if len(matches) >= MAX_GREP_LINES:
                    break
        except (OSError, ValueError, RuntimeError) as exc:
            return f"error: {exc}"
        if len(matches) >= MAX_GREP_LINES:
            matches.append(f"... truncated at {MAX_GREP_LINES} matches")
        return "\n".join(matches) if matches else "no matches"

    async def _run_command(self, command: str) -> str:
        if not command.strip():
            return "error: empty command"
        for denied in DENIED_COMMANDS:
            if denied in command:
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
