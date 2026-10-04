"""The worker agent -- a tool-using implementer for one planned task.

The worker owns the side effects of a task: it reads, writes and edits real files
inside the sandboxed workspace and reports what it changed back to the
orchestrator. All file access is delegated to :class:`furrow.llm.LLMClient`, which
confines every path to ``settings.workspace``.

:meth:`WorkerAgent.run` never raises. The orchestrator schedules the next cycle
from the results of *every* task, so an escaping exception would abort the whole
cycle; failures come back as ``WorkerResult(success=False, error=...)`` instead.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
from pydantic import BaseModel, Field

from furrow import config
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, TaskStatus
from furrow.llm import IGNORED_DIRS, LLMClient

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from furrow.config import Settings

logger = logging.getLogger(__name__)

#: Hard cap on the content handed back from ``read_file``. Source files are read
#: from the top -- imports, signatures, the region under edit -- so the head is
#: the part worth keeping.
MAX_FILE_CHARS = 40_000
#: Hard cap on the number of paths returned by ``list_files``.
MAX_LIST_FILES = 500
#: How much of the model's summary is quoted back when reporting a failure.
MAX_SUMMARY_CHARS = 500


class WorkerResult(BaseModel):
    """Outcome of one worker run.

    ``files_changed`` is the ordered, de-duplicated set of paths the worker
    touched; ``error`` carries the reason when ``success`` is False.
    """

    success: bool
    summary: str
    files_changed: list[str] = Field(default_factory=list)
    error: str | None = None


#: Anthropic-style tool schemas. ``LLMClient.agent_loop`` translates them for
#: OpenAI itself, so only the Anthropic shape is written here.
WORKER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the project. Always read a file "
        "before editing it so that old_string matches exactly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create a file or overwrite it entirely with new content. "
        "Parent directories are created as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "content": {"type": "string", "description": "The complete new file content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace an exact substring in an existing file. Prefer this over "
        "write_file for targeted changes. old_string must match the file exactly, "
        "whitespace included, and must be unique unless replace_all is true.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring a unique match.",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "List the files under a directory in the project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Workspace-relative directory. Defaults to the project root.",
                    "default": ".",
                },
            },
            "required": [],
        },
    },
]


# Self-referential phrases that mean "the model gave up". Matched against the
# head of the summary only, so a legitimate report that merely mentions a
# failure ("fixed the bug where the parser could not handle ...") is not misread
# as a refusal.
_INABILITY = re.compile(
    r"\bi\s+(?:can'?t|cannot|can\s+not|could\s+not|couldn'?t|am\s+unable|was\s+unable|"
    r"am\s+not\s+able|was\s+not\s+able|have\s+not|haven'?t|did\s+not|didn'?t)\b"
    r"|\b(?:cannot|can'?t|could\s+not|couldn'?t)\s+"
    r"(?:complete|accomplish|achieve|perform|implement|modify|change)\b"
    r"|\b(?:unable|not\s+able)\s+to\s+"
    r"(?:complete|accomplish|achieve|perform|implement|modify|change|edit|write)\b"
    r"|\bno\s+(?:changes|edits|modifications)\s+(?:were|have\s+been)\s+made\b"
    r"|\b(?:i|we)\s+(?:have\s+)?not\s+(?:made|changed|modified|edited)\s+any\b",
    re.IGNORECASE,
)


def _truncate_head(text: str, limit: int = MAX_FILE_CHARS) -> str:
    """Keep the first *limit* characters, flagging the clip explicitly."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    marker = (
        f"... [WARNING: this file is {len(text)} characters long. The first {limit} "
        f"characters are shown below and {dropped} later characters were truncated. "
        f"Use edit_file with a unique old_string instead of relying on the tail.] ...\n"
    )
    return marker + text[:limit]


def _clip(text: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    """Collapse whitespace and shorten *text* for inclusion in an error message."""
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _require_str(data: dict[str, Any], key: str) -> str:
    """Pull a required non-empty string argument out of a tool payload."""
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key!r} is required and must be a non-empty string.")
    return value


def _optional_str(data: dict[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    return value if isinstance(value, str) else str(value)


class WorkerAgent:
    """Implements a single :class:`~furrow.config.TaskModel` using LLM tools."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        # Resolve settings from the client so that whichever of the two the caller
        # supplied is authoritative, and fall back to the explicit argument or the
        # process-wide singleton. Without this, WorkerAgent(task=t) leaves
        # self.settings as None and reading worker_model raises AttributeError.
        self.settings: Settings = (
            getattr(self.client, "settings", None) or settings or config.settings
        )
        self._files_changed: list[str] = []

    # -- result mapping --------------------------------------------------

    def _record_change(self, path: str) -> None:
        """Note *path* as touched, preserving first-seen order without duplicates."""
        if path not in self._files_changed:
            self._files_changed.append(path)

    def _succeed(self, summary: str) -> WorkerResult:
        self.task.result = summary
        self.task.status = TaskStatus.COMPLETED
        self.task.status_message = None
        return WorkerResult(
            success=True,
            summary=summary,
            files_changed=list(self._files_changed),
        )

    def _fail(self, reason: str, summary: str = "") -> WorkerResult:
        self.task.status = TaskStatus.FAILED
        self.task.status_message = reason
        return WorkerResult(
            success=False,
            summary=summary or reason,
            files_changed=list(self._files_changed),
            error=reason,
        )

    # -- prompt ----------------------------------------------------------

    def _build_prompt(self) -> str:
        """Task-specific prompt; :data:`WORKER_PROMPT` is passed as the system prompt."""
        targets = ", ".join(self.task.files) if self.task.files else "choose appropriate files"
        return (
            f"Task id: {self.task.id}\n"
            f"Task: {self.task.description}\n"
            f"Target files: {targets}\n"
            "\n"
            "Every path you pass to a tool is relative to the project root, and all "
            "access is confined to that directory -- never pass an absolute path or "
            "step outside the project.\n"
            "Actually perform the edits with the tools now: read the files you are "
            "about to change first, then write or edit them. Do not merely describe "
            "the change you would make.\n"
            "When you are finished, reply with a concise summary of what you changed. "
            "If you cannot complete the task, say so plainly and explain why instead "
            "of reporting success."
        )

    # -- sandboxed file access -------------------------------------------
    #
    # These delegate to LLMClient's sandboxed helpers. The fallbacks exist so the
    # agent still behaves when it is handed a partial client (tests, stubs) -- they
    # enforce exactly the same workspace confinement.

    def _resolve_path(self, path: str | Path) -> Path:
        resolver = getattr(self.client, "resolve_path", None)
        if callable(resolver):
            return resolver(path)
        root = Path(self.settings.workspace).expanduser().resolve()
        candidate = Path(path).expanduser()
        target = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Refusing to access {str(path)!r}: outside workspace {root}")
        return target

    async def _read_text(self, path: str) -> str:
        reader = getattr(self.client, "read_file", None)
        if callable(reader):
            return str(await reader(path))
        target = self._resolve_path(path)
        async with aiofiles.open(target, encoding="utf-8", errors="replace") as handle:
            return await handle.read()

    async def _write_text(self, path: str, content: str) -> None:
        writer = getattr(self.client, "write_file", None)
        if callable(writer):
            await writer(path, content)
            return
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as handle:
            await handle.write(content)

    def _list_files(self, directory: str) -> list[str]:
        lister = getattr(self.client, "list_files", None)
        if callable(lister):
            return [str(item) for item in lister(directory)]
        root = self._resolve_path(directory)
        if not root.is_dir():
            return []
        found: list[str] = []
        for entry in sorted(root.rglob("*")):
            parts = entry.relative_to(root).parts
            if any(part in IGNORED_DIRS for part in parts):
                continue
            if entry.is_file():
                found.append(str(Path(*parts)))
        return sorted(found)

    # -- tool implementations --------------------------------------------

    async def _tool_read_file(self, data: dict[str, Any]) -> str:
        path = _require_str(data, "path")
        text = await self._read_text(path)
        return _truncate_head(text)

    async def _tool_write_file(self, data: dict[str, Any]) -> str:
        path = _require_str(data, "path")
        content = _optional_str(data, "content")
        await self._write_text(path, content)
        self._record_change(path)
        lines = content.count("\n") + 1 if content else 0
        return f"Wrote {path} ({len(content)} characters, {lines} lines)."

    async def _tool_edit_file(self, data: dict[str, Any]) -> str:
        path = _require_str(data, "path")
        old_string = _require_str(data, "old_string")
        new_string = _optional_str(data, "new_string")
        replace_all = bool(data.get("replace_all", False))

        text = await self._read_text(path)
        occurrences = text.count(old_string)
        if occurrences == 0:
            raise ValueError(
                f"old_string was not found in {path}. Read the file again and copy the "
                "exact text to replace, including indentation and trailing newlines."
            )
        if occurrences > 1 and not replace_all:
            raise ValueError(
                f"old_string occurs {occurrences} times in {path}. Include more "
                "surrounding context so the match is unique, or pass replace_all=true."
            )
        if replace_all:
            updated = text.replace(old_string, new_string)
            replaced = occurrences
        else:
            updated = text.replace(old_string, new_string, 1)
            replaced = 1
        await self._write_text(path, updated)
        self._record_change(path)
        return f"Replaced {replaced} occurrence(s) in {path}."

    async def _tool_list_files(self, data: dict[str, Any]) -> str:
        directory = _optional_str(data, "directory", ".") or "."
        files = await asyncio.to_thread(self._list_files, directory)
        if not files:
            return f"No files found under {directory!r}."
        shown = files[:MAX_LIST_FILES]
        listing = "\n".join(shown)
        if len(files) > MAX_LIST_FILES:
            listing += f"\n... and {len(files) - MAX_LIST_FILES} more file(s) not shown."
        return f"{len(files)} file(s) under {directory!r}:\n{listing}"

    async def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch one tool call. Never raises -- the model gets an error string."""
        handler: Callable[[dict[str, Any]], Awaitable[str]] | None = getattr(
            self, f"_tool_{name}", None
        )
        if handler is None:
            available = ", ".join(tool["name"] for tool in WORKER_TOOLS)
            return f"Error: unknown tool {name!r}. Available tools: {available}."
        payload = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
        try:
            return await handler(payload)
        except Exception as exc:  # reported back so the model can retry
            logger.warning("Worker tool %s failed: %s", name, exc)
            return f"Error: {type(exc).__name__}: {exc}"

    # -- entry point ------------------------------------------------------

    async def run(self) -> WorkerResult:
        """Implement the task. Always returns a :class:`WorkerResult`; never raises."""
        try:
            self.task.status = TaskStatus.RUNNING
            self.task.status_message = None
            self._files_changed.clear()
            summary = str(
                await self.client.agent_loop(
                    self._build_prompt(),
                    WORKER_TOOLS,
                    self._execute_tool,
                    system=WORKER_PROMPT,
                    model=self.settings.worker_model,
                )
                or ""
            ).strip()

            if not summary:
                return self._fail("The worker model returned no summary.")
            if _INABILITY.search(summary[:MAX_SUMMARY_CHARS]):
                return self._fail(
                    f"The worker reported it could not complete the task: {_clip(summary)}",
                    summary=summary,
                )
            return self._succeed(summary)
        except Exception as exc:
            logger.exception("Worker failed for task %s", self.task.id)
            return self._fail(f"{type(exc).__name__}: {exc}")
