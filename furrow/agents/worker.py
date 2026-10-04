"""Worker agent: turns one planned task into real files on disk.

The worker is the only stage that mutates the workspace, so it is the stage
where a hallucinated path or an unparsable reply would do damage. Two
mechanisms keep that contained: every write goes through
:meth:`furrow.llm.LLMClient.write_file`, which is sandboxed to
``settings.workspace``, and every individual write failure is recorded in
:attr:`WorkerAgent.errors` instead of aborting the remaining files. A reply the
model could not format as the documented JSON degrades to a plain summary with
zero files written rather than raising, because losing the whole task over one
stray token is worse than reporting no change.

Example:
    ```python
    worker = WorkerAgent(task=TaskModel(id="1", description="add src/a.py"))
    summary = await worker.run()
    if worker.errors:
        ...
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from furrow.agents.parsing import parse_json_response
from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM_PROMPT
from furrow.config import TaskModel, get_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from pathlib import Path

    from furrow.config import Settings

__all__ = ["FileEdit", "WorkerAgent", "WorkerResponse"]

MAX_CONTEXT_FILES: int = 8
"""Most task files whose existing contents are echoed into the prompt."""

MAX_CONTEXT_CHARS: int = 4000
"""Per-file cap on the contents echoed into the prompt."""


class FileEdit(BaseModel):
    """One file the worker wants written to the workspace.

    Attributes:
        path: Workspace-relative path; absolute or escaping paths are refused
            by :meth:`furrow.llm.LLMClient.write_file`.
        content: Complete new contents of the file.
    """

    path: str
    content: str = ""


class WorkerResponse(BaseModel):
    """Structured reply expected from the worker model.

    Attributes:
        summary: One paragraph describing what changed.
        files: Every file to write; empty when no code change was needed.
    """

    summary: str = ""
    files: list[FileEdit] = Field(default_factory=list)


class WorkerAgent:
    """Executes a single :class:`~furrow.config.TaskModel` against the workspace.

    Attributes:
        task: The task being executed.
        settings: Resolved configuration: the explicit ``settings`` argument
            when one is given, otherwise the client's, otherwise the
            process-global :func:`~furrow.config.get_settings` singleton.
        client: The LLM client used for the completion and for every file write.
        files_written: Workspace-relative paths successfully written by the last
            :meth:`run`.
        errors: Human-readable problems encountered by the last :meth:`run`,
            covering malformed replies and rejected or failed writes.
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Bind a task to a client.

        Args:
            task: The task to implement.
            client: Client to reuse. Its ``settings`` are consulted only
                when no ``settings`` argument is supplied.
            settings: Configuration that takes precedence over the client's,
                and that builds the client when none is given.
        """
        self.settings = settings or (client.settings if client is not None else get_settings())
        self.task = task
        self.client = client or LLMClient(settings=self.settings)
        self.files_written: list[str] = []
        self.errors: list[str] = []

    async def run(self) -> str:
        """Implement the task, writing the returned files to the workspace.

        Files are written as a side effect through the sandboxed client. A
        reply that is not the documented JSON is treated as a plain summary, so
        this method never fails to return because of malformed model output;
        genuinely fatal problems (a failed LLM request) still propagate as
        :class:`~furrow.llm.LLMError`.

        Returns:
            The model's summary of the work, or its raw text when the reply
            could not be parsed as JSON.

        Raises:
            furrow.llm.LLMError: If the completion request fails.
        """
        response = await self.client.complete(
            self.build_prompt(),
            system=WORKER_SYSTEM_PROMPT,
            model=self.client.settings.worker_model,
        )
        try:
            payload = parse_json_response(response, WorkerResponse, "worker response")
        except ValueError as exc:
            self.errors.append(str(exc))
            return response.strip()
        summary = payload.summary.strip() or response.strip()
        await self._write_files(payload.files)
        return summary

    def build_prompt(self) -> str:
        """Assemble the worker prompt for the bound task.

        Returns:
            The prompt text, containing the task id, description, files,
            dependencies, any result carried over from an earlier attempt, and
            the current contents of the task's files where they are readable.
        """
        task = self.task
        lines = [
            WORKER_PROMPT,
            "",
            f"Task id: {task.id}",
            f"Task description: {task.description}",
            f"Files in scope: {', '.join(task.files) if task.files else 'your choice'}",
            f"Dependencies: {', '.join(task.dependencies) if task.dependencies else 'none'}",
        ]
        if task.result:
            lines.append(f"Result of a previous attempt: {task.result}")
        context = self._file_context()
        if context:
            lines.extend(["", "Current contents of the files in scope:", context])
        return "\n".join(lines)

    async def _write_files(self, files: list[FileEdit]) -> None:
        """Write every requested file, recording per-file failures.

        Args:
            files: The file edits extracted from the worker reply.
        """
        for edit in files:
            try:
                await self.client.write_file(edit.path, edit.content)
            except ValueError as exc:
                self.errors.append(f"refused to write {edit.path!r}: {exc}")
            except OSError as exc:
                self.errors.append(f"could not write {edit.path!r}: {exc}")
            else:
                self.files_written.append(edit.path)

    def _file_context(self) -> str:
        """Read the task's files so the model can rewrite them in full.

        Unreadable or oversized files are described rather than inlined, since
        the protocol requires complete file contents in the reply.

        Returns:
            A block of ``path:``-prefixed excerpts, empty when nothing could be
            read.
        """
        chunks: list[str] = []
        for path in self.task.files[:MAX_CONTEXT_FILES]:
            try:
                text = self._read_text(self.client.resolve_path(path))
            except ValueError as exc:
                chunks.append(f"{path}: <unavailable: {exc}>")
                continue
            if len(text) > MAX_CONTEXT_CHARS:
                chunks.append(
                    f"{path}: <{len(text)} characters, too large to inline; "
                    f"first {MAX_CONTEXT_CHARS} characters follow>\n{text[:MAX_CONTEXT_CHARS]}"
                )
            else:
                chunks.append(f"{path}:\n{text}")
        return "\n\n".join(chunks)

    def _read_text(self, path: Path) -> str:
        """Read a resolved workspace file synchronously.

        The worker reads prompt context only, so it never needs a second event
        loop; failures are handled by the caller.

        Args:
            path: Absolute path inside the workspace.

        Returns:
            The file's text, or a placeholder if it cannot be decoded.
        """
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"<unavailable: {exc}>"
