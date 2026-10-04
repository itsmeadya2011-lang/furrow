"""Worker agent: implement one planned task by actually writing files into the workspace.

This module is the difference between furrow being an autonomous coding agent and being a
chat client that summarises code. The old :meth:`WorkerAgent.run` returned the model's prose
summary straight out of :meth:`furrow.llm.LLMClient.complete`: it never read the project,
never wrote a byte, and left :meth:`furrow.llm.LLMClient.write_file` as dead code, so a full
run of the orchestrator produced zero code changes while appearing to succeed.

The worker therefore has three jobs:

* build a grounded prompt (the task, the files it may touch with their current contents, and
  a short workspace listing);
* parse the model's ``WorkerResult`` and apply every requested change to disk;
* guard the path. The paths in a model response are attacker-adjacent data, so
  :meth:`WorkerAgent._safe_target` refuses anything absolute, containing ``..``, or landing
  outside the workspace.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import get_settings
from furrow.jsonutil import parse_model_json, truncate
from furrow.llm import LLMClient
from furrow.models import FileChange, TaskModel, TaskStatus, WorkerResult

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["MAX_CONTEXT_FILES", "MAX_FILE_CONTEXT_BYTES", "WorkerAgent"]

#: Cap on the Python files listed to the model as workspace context.
MAX_CONTEXT_FILES = 40

#: Byte budget per task file included in the prompt. The default 200 kB of
#: ``LLMClient.read_file`` would let two or three large files eat the whole completion
#: budget, so the listing is capped well below it.
MAX_FILE_CONTEXT_BYTES = 20_000

#: Cap on how many of ``task.files`` are read into the prompt, in case a planner lists a
#: whole package.
MAX_TASK_FILES_READ = 10


class WorkerAgent:
    """Execute a single :class:`~furrow.models.TaskModel` against the workspace."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Bind the worker to `task`.

        Settings are resolved eagerly (``settings=None`` -> :func:`get_settings`), so
        ``WorkerAgent(task)`` never stores ``None`` and never raises ``AttributeError``
        later on.
        """
        self.task = task
        self.settings: Settings = settings if settings is not None else get_settings()
        self.client = client or LLMClient(self.settings)
        self.workspace: Path = self.settings.workspace_path()
        #: Workspace-relative POSIX paths of the files this worker actually wrote.
        self.written_files: list[str] = []
        #: One human-readable message per change that was rejected or failed to apply.
        self.errors: list[str] = []

    async def run(self) -> str:
        """Execute the task and return a human-readable summary.

        Every ``FileChange`` in the model's response is applied to the workspace before this
        returns: ``"write"`` goes through :meth:`LLMClient.write_file`, ``"delete"`` unlinks
        the file.

        Returns:
            The model's ``summary``, plus a ``Wrote:`` line naming the files written, a
            ``Deleted:`` line when files were removed, and an ``Errors:`` section listing
            every change that was rejected or failed. The same text is stored in
            ``self.task.result``.

        Raises:
            ValueError: if the model response cannot be parsed as a
                :class:`~furrow.models.WorkerResult`, or if the response asked for at least
                one change and *none* of them could be applied. A partially applied plan is
                not an error: it is reported through ``self.errors`` so the orchestrator can
                show what actually landed.
        """
        response = await self.client.complete(
            await self._build_prompt(),
            system=WORKER_PROMPT,
            model=self.settings.models()["worker"],
        )
        try:
            result = parse_model_json(response, WorkerResult)
        except ValueError as exc:
            message = f"worker returned output that is not a WorkerResult: {exc}"
            self.task.status = TaskStatus.FAILED
            self.task.result = message
            raise ValueError(f"{message}; raw response: {truncate(response, 500)}") from exc

        requested = len(result.files)
        written, deleted = await self._apply(result.files)
        if requested and not (written or deleted):
            message = "no requested change could be applied: " + "; ".join(self.errors)
            self.task.status = TaskStatus.FAILED
            self.task.result = message
            raise ValueError(message)

        summary = self._summary(result, written, deleted)
        self.task.status = TaskStatus.FAILED if self.errors else TaskStatus.COMPLETED
        self.task.result = summary
        return summary

    async def _apply(self, changes: list[FileChange]) -> tuple[list[str], list[str]]:
        """Apply `changes` in order; return the relative paths written and deleted.

        Every failure is appended to :attr:`errors` rather than raised, so one bad path in a
        five-file response does not throw away the four good writes.
        """
        written: list[str] = []
        deleted: list[str] = []
        for change in changes:
            try:
                target = self._safe_target(change.path)
            except ValueError as exc:
                self.errors.append(f"rejected {change.path!r}: {exc}")
                continue
            relative = target.relative_to(self.workspace).as_posix()
            action = (change.action or "write").strip().lower()
            if action == "write":
                try:
                    await self.client.write_file(target, change.content)
                except OSError as exc:
                    self.errors.append(f"failed to write {relative}: {exc}")
                    continue
                written.append(relative)
                self.written_files.append(relative)
            elif action == "delete":
                try:
                    # missing_ok: deleting a file the model thought existed is a no-op, not
                    # an error, because the end state is what the task asked for.
                    target.unlink(missing_ok=True)
                except OSError as exc:
                    self.errors.append(f"failed to delete {relative}: {exc}")
                    continue
                deleted.append(relative)
            else:
                self.errors.append(
                    f"unknown action {change.action!r} for {relative}; "
                    "expected 'write' or 'delete'"
                )
        return written, deleted

    def _safe_target(self, relative: str) -> Path:
        """Resolve `relative` inside the workspace, rejecting absolute paths, `..`
        escapes and symlinks that leave the workspace. Raises ValueError otherwise.

        Path traversal guard: every path in a model response is untrusted input, and the
        worker runs with the operator's privileges. A response containing ``../../.ssh/
        authorized_keys`` must not be able to write outside the project. The check is
        deliberately belt-and-braces -- reject the syntactically dangerous forms *and*
        resolve the real path (which follows symlinks) and re-check containment, because a
        perfectly innocent-looking ``link.py`` inside the workspace may itself be a symlink
        to ``/etc/passwd``.
        """
        candidate = relative.strip()
        # Normalise Windows separators first so "..\\x.py" is caught by the checks below.
        candidate = candidate.replace("\\", "/")
        if not candidate:
            raise ValueError("empty path")
        if candidate.startswith("~"):
            raise ValueError("home-relative paths are not allowed")
        windows = PureWindowsPath(candidate)
        if windows.drive or windows.root or windows.is_absolute():
            raise ValueError("absolute paths are not allowed; use a workspace-relative path")
        if PurePosixPath(candidate).is_absolute() or Path(candidate).is_absolute():
            raise ValueError("absolute paths are not allowed; use a workspace-relative path")
        if ".." in PurePosixPath(candidate).parts:
            raise ValueError("parent-directory traversal ('..') is not allowed")
        try:
            target = (self.workspace / candidate).resolve()
        except (OSError, RuntimeError) as exc:
            # A symlink loop raises RuntimeError on 3.10 and OSError on 3.13+; a path we
            # cannot resolve is a path we are not willing to write.
            raise ValueError(f"could not be resolved: {exc}") from exc
        if not target.is_relative_to(self.workspace):
            raise ValueError(f"resolves to {target}, which is outside the workspace")
        return target

    async def _build_prompt(self) -> str:
        """Compose the user message: task, workspace overview and current file contents."""
        try:
            listing = self.client.list_files(
                self.workspace, pattern="*.py", max_entries=MAX_CONTEXT_FILES
            )
        except OSError as exc:  # racing or unreadable tree; the task does not depend on it
            listing = []
            overview = f"<workspace listing unavailable: {exc}>"
        else:
            overview = "\n".join(f"  {path}" for path in listing) or "  <no Python files>"

        parts = [
            f"Workspace root: {self.workspace}",
            f"Python files in the workspace (relative paths, up to {MAX_CONTEXT_FILES}):",
            overview,
            "",
            f"Task id: {self.task.id}",
            f"Task: {self.task.description}",
        ]
        if self.task.dependencies:
            parts.append(f"Depends on: {', '.join(self.task.dependencies)}")
        parts += await self._file_context()
        return "\n".join(parts)

    async def _file_context(self) -> list[str]:
        """Current contents of the task's files, so the model edits rather than invents."""
        wanted = self.task.files[:MAX_TASK_FILES_READ]
        skipped = len(self.task.files) - len(wanted)
        parts: list[str] = []
        for name in wanted:
            parts += ["", f"--- current contents of {name} ---"]
            try:
                target = self._safe_target(name)
            except ValueError as exc:
                parts.append(f"<unreadable: {exc}>")
                continue
            try:
                parts.append(await self.client.read_file(target, max_bytes=MAX_FILE_CONTEXT_BYTES))
            except OSError as exc:
                parts.append(f"<unreadable: {exc}>")
        if not wanted:
            parts += ["", "No files were named for this task; pick the paths you create."]
        elif skipped > 0:
            parts += ["", f"({skipped} further file(s) were named but are not shown)"]
        return parts

    def _summary(self, result: WorkerResult, written: list[str], deleted: list[str]) -> str:
        """Render the worker's outcome: model summary, files touched, and any errors."""
        lines = [result.summary.strip() or f"(the worker returned no summary for {self.task.id})"]
        if written:
            lines.append(f"Wrote: {', '.join(written)}")
        if deleted:
            lines.append(f"Deleted: {', '.join(deleted)}")
        if self.errors:
            lines.append("Errors:")
            lines += [f"- {error}" for error in self.errors]
        if result.notes.strip():
            lines += ["Notes:", result.notes.strip()]
        return "\n".join(lines)
