"""Execute one task: ground the model in real file contents, then apply its file writes.

The worker is the only agent that touches the filesystem, so it is also where a model reply
gets constrained. Two rules hold no matter what the model says: writes are confined to
``settings.workspace``, and a reply that carries no parseable description of file changes
writes nothing at all rather than reporting a success that never happened.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger("furrow.agents.worker")

#: Characters of an existing file shown to the model; longer files are truncated.
MAX_FILE_CONTENT_CHARS = 8_000

#: Characters of the model's own summary echoed back to the orchestrator.
MAX_SUMMARY_CHARS = 600

#: The reply contract, appended last so it has the model's attention. JSON rather than a
#: fenced code block with a path header: it is unambiguous to parse and it keeps the
#: content of a file distinct from the prose around it.
FILE_CHANGE_SHAPE = """Reply with a single JSON object and nothing else (no prose before or
after it, no markdown code fences) with this shape:
{
  "summary": "one sentence describing what you changed",
  "files": [
    {"path": "relative/path/from/workspace/root.py", "content": "the complete new file contents"}
  ]
}

Rules for the reply:
- "content" is the complete contents of the file after your change, never a diff or a fragment.
- Only include files you actually changed; return "files": [] if no change is needed.
- Paths are relative to the workspace root. Absolute paths and ".." are rejected.
"""


class WorkerAgent:
    """Implement a single :class:`~furrow.config.TaskModel` by applying model file writes."""

    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        # Same explicit fallback as the planner: a bare WorkerAgent() must still get a
        # configured client and a non-None ``settings``.
        self.client = (
            client if client is not None else LLMClient(settings=settings or default_settings)
        )
        self.settings: Settings = (
            getattr(self.client, "settings", None) or settings or default_settings
        )

    async def run(self) -> str:
        """Apply the task's file changes and return a summary string for the orchestrator.

        Never raises for an unusable reply: no parseable JSON means no writes, and the
        returned string says exactly that.
        """
        context = await self._current_contents()
        prompt = self._build_prompt(context)
        response = await self.client.complete(prompt, model=self.settings.worker_model)

        try:
            summary, writes = self._parse_changes(response)
        except ValueError as exc:
            logger.warning("Task %r: no usable file changes in the model reply: %s", self.task.id, exc)
            return f"Task {self.task.id}: no file changes ({exc})."

        if not writes:
            detail = summary or "the model returned an empty file list"
            logger.info("Task %r: %s", self.task.id, detail)
            return f"Task {self.task.id}: no file changes ({detail})."

        written: list[str] = []
        refused: list[str] = []
        failed: list[str] = []
        for path, content in writes:
            try:
                target = self._resolve(path)
            except ValueError as exc:
                refused.append(f"{path!r} ({exc})")
                logger.warning("Task %r: refusing to write %r: %s", self.task.id, path, exc)
                continue
            try:
                await self.client.write_file(target, content)
            except OSError as exc:
                failed.append(f"{path!r} ({exc})")
                logger.warning("Task %r: could not write %r: %s", self.task.id, path, exc)
                continue
            written.append(path)

        parts = [f"Task {self.task.id}: wrote {len(written)} of {len(writes)} requested file(s)"]
        if written:
            parts.append(", ".join(written))
        if refused:
            parts.append(f"refused {len(refused)} unsafe path(s): {'; '.join(refused)}")
        if failed:
            parts.append(f"{len(failed)} write error(s): {'; '.join(failed)}")
        if summary:
            parts.append(f"summary: {summary}")
        return " | ".join(parts)

    # ---------------------------------------------------------------------- prompt --

    def _build_prompt(self, context: str) -> str:
        """Assemble the worker prompt, ending with the reply contract."""
        files = ", ".join(self.task.files) if self.task.files else "any file you judge necessary"
        return (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n\n"
            f"Current contents of those files:\n{context}\n\n"
            f"{FILE_CHANGE_SHAPE}"
        )

    async def _current_contents(self) -> str:
        """Return the existing content of the assigned files, bounded per file.

        Files that do not exist yet are the normal case for new modules, so they are
        reported as such rather than treated as an error.
        """
        if not self.task.files:
            return "(no files were assigned to this task; choose what you need)"
        blocks: list[str] = []
        for path in self.task.files:
            try:
                target = self._resolve(path)
            except ValueError as exc:
                logger.warning("Task %r: not showing %r: %s", self.task.id, path, exc)
                blocks.append(f"--- {path} --- (skipped: {exc})")
                continue
            try:
                content = await self.client.read_file(target)
            except (FileNotFoundError, NotADirectoryError):
                logger.info("Task %r: %r does not exist yet.", self.task.id, path)
                blocks.append(f"--- {path} --- (does not exist yet; create it)")
                continue
            except OSError as exc:
                logger.warning("Task %r: could not read %r: %s", self.task.id, path, exc)
                blocks.append(f"--- {path} --- (unreadable: {exc})")
                continue
            if len(content) > MAX_FILE_CONTENT_CHARS:
                content = content[:MAX_FILE_CONTENT_CHARS] + "\n... (truncated)"
            blocks.append(f"--- {path} ---\n{content}")
        return "\n\n".join(blocks)

    # --------------------------------------------------------------------- parsing --

    def _parse_changes(self, response: str) -> tuple[str, list[tuple[str, str]]]:
        """Return ``(summary, [(path, content)])`` described by the model's JSON reply.

        Raises:
            ValueError: If the reply carries no JSON object at all. An object that simply
                declares no changes is not an error and returns an empty ``writes`` list.
        """
        try:
            data = extract_json(response)
        except ValueError as exc:
            raise ValueError(f"the model reply contained no JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            # The ValueError is deliberate: an unusable model reply is a value problem
            # for the caller, not a bug.
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")  # noqa: TRY004

        summary = str(data.get("summary") or "").strip()[:MAX_SUMMARY_CHARS]
        files = data.get("files")
        if files is None:
            return summary, []
        if not isinstance(files, list):
            raise ValueError(f"'files' must be a list, got {type(files).__name__}")  # noqa: TRY004

        writes: list[tuple[str, str]] = []
        for entry in files:
            if not isinstance(entry, Mapping):
                logger.warning("Task %r: dropping non-object file entry %r", self.task.id, entry)
                continue
            path = str(entry.get("path") or "").strip()
            content = entry.get("content")
            if not path:
                logger.warning("Task %r: dropping file entry with no path", self.task.id)
                continue
            if not isinstance(content, str):
                logger.warning(
                    "Task %r: dropping %r: 'content' must be a string, got %s",
                    self.task.id,
                    path,
                    type(content).__name__,
                )
                continue
            writes.append((path, content))
        return summary, writes

    # ------------------------------------------------------------------------ paths --

    def _resolve(self, path: str) -> Path:
        """Resolve a model supplied path inside the workspace or refuse it.

        Raises:
            ValueError: If the path is absolute, contains ``..``, or escapes the
                workspace through a symlink.
        """
        if not path:
            raise ValueError("empty path")
        if path.startswith(("/", "\\", "~")) or os.path.isabs(path):
            raise ValueError("absolute paths are not allowed")
        if len(path) > 1 and path[1] == ":":
            raise ValueError("drive-qualified paths are not allowed")
        # Both separator conventions are checked: a backslash is an ordinary character on
        # POSIX, so "..\..\x" would otherwise slip past a POSIX-only check as a single
        # (harmless on POSIX, escaping on Windows) filename component.
        if ".." in PurePosixPath(path).parts or ".." in PureWindowsPath(path).parts:
            raise ValueError("'..' traversal is not allowed")
        # Resolving both sides also catches a symlink inside the workspace that points out
        # of it, which a purely lexical check would miss.
        root = Path(self.settings.workspace).resolve()
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            raise ValueError("path escapes the workspace")
        return target