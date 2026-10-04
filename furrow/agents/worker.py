"""Worker agent: the component that actually mutates files in the workspace.

Protocol
--------
The worker asks the model for edits in an explicit, machine-parseable wire format --
one block per file holding the COMPLETE new content -- and applies them itself::

    <furrow:file path="src/foo.py">
    ...complete new content...
    </furrow:file>
    <furrow:delete path="src/obsolete.py"/>

``WorkerAgent.run()`` keeps its original contract (async, returns a summary ``str``
that the orchestrator stores on ``TaskModel.result``) and never raises: LLM and
filesystem problems are captured in ``WorkerAgent.error``. Every path actually
mutated on disk is reported in ``WorkerAgent.changed_files``, and the subset that
was deleted is also listed in ``WorkerAgent.deleted_files``. Writes are confined to
the workspace root; paths resolving outside of it are refused.

Repair cycles
-------------
The optional ``failures`` constructor argument carries the failure text of the
previous cycle's test run. When it is non-empty the worker is on a *repair* cycle:
it prompts with ``FIX_PROMPT`` instead of ``WORKER_PROMPT`` and pastes the failures
into the prompt with an explicit instruction to repair the root cause rather than
delete a test or weaken an assertion. Without it the worker behaves exactly as
before -- the generic implementation framing and the same prompt body -- so every
existing call site keeps working unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import FIX_PROMPT, WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from collections.abc import Sequence

    from furrow.config import Settings

IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".pytest_cache",
    }
)

MAX_TREE_ENTRIES = 200
MAX_FILE_CHARS = 4000
MAX_PROMPT_CHARS = 60_000
MAX_SUMMARY_CHARS = 800
TRUNCATION_MARKER = "\n... [truncated]"

_EDIT_RE = re.compile(
    r"<furrow:file\b(?P<file_attrs>[^>]*)>(?P<content>.*?)<\s*/\s*furrow:file\s*>"
    r"|<furrow:delete\b(?P<delete_attrs>[^>]*)",
    re.DOTALL | re.IGNORECASE,
)
_PATH_ATTR_RE = re.compile(
    r"""\bpath\s*=\s*(?:(["'])(?P<quoted>.*?)\1|(?P<bare>[^\s"'<>`]+))""",
    re.DOTALL,
)


def _extract_path(attrs: str) -> str | None:
    """Return the ``path`` attribute of an opening tag, or None when absent."""
    match = _PATH_ATTR_RE.search(attrs)
    if match is None:
        return None
    raw = match.group("quoted") if match.group("quoted") is not None else match.group("bare")
    return raw.strip() or None


def _normalize_content(content: str) -> str:
    """Normalize block content to LF endings, dropping the newline that frames the tag.

    Everything else is kept verbatim, including a trailing newline, so written files
    keep their final line break.
    """
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    text = text.removeprefix("\n")
    return text


def _is_ignored(path: Path) -> bool:
    """True when any path component is an ignored directory or byte-compiled file."""
    return any(part in IGNORED_DIRS or part.endswith(".pyc") for part in path.parts)


def _safe_target(root: Path, raw_path: str) -> Path | None:
    """Resolve ``raw_path`` inside ``root``; None when it escapes or is ignored."""
    cleaned = raw_path.strip()
    if not cleaned:
        return None
    try:
        resolved = (root / cleaned).resolve()
    except OSError:
        return None
    if resolved == root or root not in resolved.parents:
        return None
    if _is_ignored(resolved.relative_to(root)):
        return None
    return resolved


def _prose_summary(text: str) -> str:
    """Return the model's prose, with all edit blocks removed."""
    chunks: list[str] = []
    cursor = 0
    for match in _EDIT_RE.finditer(text):
        start, end = match.span()
        if start > cursor:
            chunks.append(text[cursor:start])
        cursor = max(cursor, end)
    chunks.append(text[cursor:])
    prose = " ".join(chunk.strip() for chunk in chunks if chunk.strip())
    prose = prose.replace("```", " ").strip()
    if len(prose) > MAX_SUMMARY_CHARS:
        prose = prose[:MAX_SUMMARY_CHARS].rstrip() + " ..."
    return prose


class WorkerAgent:
    """Implements a task by prompting the model for edits and applying them.

    ``failures`` is the optional repair-cycle context: the failure lines reported by
    the previous cycle's test run. When it is non-empty the prompt is built from
    ``FIX_PROMPT`` (see the module docstring); when it is omitted or empty the
    worker uses ``WORKER_PROMPT`` and behaves exactly as before.
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        *,
        failures: Sequence[str] | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        #: Failure lines of the previous cycle; empty on a normal (non-repair) cycle.
        self.failures: list[str] = [line for line in (failures or ()) if line.strip()]
        self.changed_files: list[str] = []
        self.deleted_files: list[str] = []
        self.error: str | None = None
        self._notes: list[str] = []

    async def run(self) -> str:
        """Prompt the model, apply its edits, and return a summary. Never raises."""
        self.changed_files = []
        self.deleted_files = []
        self.error = None
        self._notes = []
        try:
            workspace = Path(self.client.settings.workspace)
            prompt = await self._build_prompt(workspace)
            model = self.client.settings.model_for("worker")
            response = await self.client.complete(prompt, model=model)
        except Exception as exc:  # noqa: BLE001 - run() must never raise, by contract
            return self._fail(f"the model request failed: {exc!r}")
        try:
            await self._apply_edits(response, workspace)
        except Exception as exc:  # noqa: BLE001 - run() must never raise, by contract
            self._fail(f"applying the requested edits failed: {exc!r}")
        return self._render_summary(response)

    async def _build_prompt(self, workspace: Path) -> str:
        """Compose the protocol, the task, the project tree and the file contents."""
        root = workspace.resolve()
        priority = [await self._file_section(root, raw) for raw in self.task.files]
        return self._assemble(self._header(workspace), priority, [self._project_tree(root)])

    def _header(self, workspace: Path) -> str:
        allowed = ", ".join(self.task.files) if self.task.files else "any path in the workspace"
        lines = [
            self._base_prompt(),
            "",
            f"Task id: {self.task.id}",
            f"Task: {self.task.description}",
        ]
        if self.failures:
            lines += ["", self._repair_section()]
        lines += [
            "",
            f"Workspace root: {workspace}",
            "Writes are restricted to paths inside the workspace root.",
            f"Allowed paths: {allowed}",
        ]
        if self.task.files:
            lines += ["", "Current contents of the files you are expected to touch:"]
        return "\n".join(lines)

    def _base_prompt(self) -> str:
        """``FIX_PROMPT`` on a repair cycle, ``WORKER_PROMPT`` otherwise."""
        return FIX_PROMPT if self.failures else WORKER_PROMPT

    def _repair_section(self) -> str:
        """The failure context of a repair cycle, plus the rule for acting on it."""
        body = "\n".join(f"  - {line}" for line in self.failures)
        return (
            "This is a REPAIR cycle: the previous cycle's test run FAILED and your edits must "
            f"repair these failures:\n{body}\n"
            "Repair the root cause in the implementation. Do NOT delete a test, weaken an "
            "assertion, loosen a type check or skip a case to make the suite pass."
        )

    def _project_tree(self, root: Path) -> str:
        """List the workspace files, skipping ignored paths, capped at MAX_TREE_ENTRIES."""
        try:
            entries = self.client.list_files(root)
        except OSError as exc:
            return f"Project tree unavailable ({exc})."
        visible = [path for path in entries if not _is_ignored(Path(path))]
        shown = sorted(visible)[:MAX_TREE_ENTRIES]
        note = ""
        if len(visible) > len(shown):
            note = f", showing {len(shown)}/{len(visible)} (truncated)"
        lines = [f"Project tree ({len(shown)} files{note}), relative to the workspace root:"]
        if not shown:
            lines.append("(the workspace is empty)")
        return "\n".join(lines + [f"- {path}" for path in shown])

    async def _file_section(self, root: Path, raw_path: str) -> str:
        """Render one task file: its current content, or a note that it is missing."""
        label = f"--- {raw_path} ---"
        target = _safe_target(root, raw_path)
        if target is None:
            return f"{label}\n(refused as a context path: outside the workspace or ignored)"
        content = await self._read_existing(target)
        if content is None:
            return f"{label}\n(does not exist — create it)"
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + TRUNCATION_MARKER
        return f"{label}\n```{target.suffix}\n{content}\n```"

    @staticmethod
    def _assemble(head: str, priority: list[str], extra: list[str]) -> str:
        """Join prompt sections under MAX_PROMPT_CHARS, dropping the least useful first."""
        parts = [head]
        budget = max(MAX_PROMPT_CHARS - len(head), 0)
        for index, section in enumerate(priority):
            if len(section) <= budget:
                parts.append(section)
                budget -= len(section)
                continue
            parts.append(section[: max(budget - len(TRUNCATION_MARKER), 0)] + TRUNCATION_MARKER)
            budget = 0
            dropped = len(priority) - index - 1
            if dropped:
                parts.append(f"[{dropped} further file(s) omitted: prompt size limit reached]")
            break
        for section in extra:
            if budget <= 0:
                parts.append("[project tree omitted: prompt size limit reached]")
                break
            if len(section) <= budget:
                parts.append(section)
                budget -= len(section)
                continue
            parts.append(section[:budget] + TRUNCATION_MARKER)
            break
        return "\n".join(parts)

    async def _apply_edits(self, response: str, workspace: Path) -> None:
        """Parse the response and write/delete every requested path, in response order."""
        root = workspace.resolve()
        blocks = 0
        for match in _EDIT_RE.finditer(response):
            blocks += 1
            attrs = match.group("file_attrs")
            if attrs is not None:
                target = self._resolve(root, attrs)
                if target is not None:
                    await self._write(root, target, _normalize_content(match.group("content")))
            else:
                target = self._resolve(root, match.group("delete_attrs"))
                if target is not None:
                    self._delete(root, target)
        if blocks == 0:
            self._fail(
                "the response held no <furrow:file> or <furrow:delete> block; "
                "no edit could be applied"
            )
        elif not self.changed_files and self.error is None:
            self._fail("every requested edit was a no-op; no file content changed")

    def _resolve(self, root: Path, attrs: str) -> Path | None:
        """Validate an opening tag's path attribute, recording any refusal in self.error."""
        raw_path = _extract_path(attrs)
        if raw_path is None:
            self._fail(f"ignored an edit block without a usable path attribute: {attrs.strip()!r}")
            return None
        target = _safe_target(root, raw_path)
        if target is None:
            self._fail(
                f"refused to touch {raw_path!r}: it resolves outside the workspace "
                "root or is an ignored path"
            )
            return None
        return target

    async def _write(self, root: Path, target: Path, content: str) -> None:
        relative = target.relative_to(root).as_posix()
        if await self._read_existing(target) == content:
            self._notes.append(f"{relative} was already up to date and was not rewritten")
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(target, content)
        except OSError as exc:
            self._fail(f"could not write {relative}: {exc}")
            return
        self._record(relative)

    def _delete(self, root: Path, target: Path) -> None:
        relative = target.relative_to(root).as_posix()
        if not target.is_file():
            self._notes.append(f"{relative} was requested for deletion but does not exist")
            return
        try:
            target.unlink()
        except OSError as exc:
            self._fail(f"could not delete {relative}: {exc}")
            return
        self._record(relative)
        self.deleted_files.append(relative)

    async def _read_existing(self, target: Path) -> str | None:
        """Read a file for comparison purposes; None when absent or unreadable."""
        if not target.is_file():
            return None
        try:
            return await self.client.read_file(target)
        except OSError:
            return None

    def _record(self, relative: str) -> None:
        if relative not in self.changed_files:
            self.changed_files.append(relative)

    def _fail(self, message: str) -> str:
        """Record an error and return it as a standalone summary."""
        self.error = f"{self.error}; {message}" if self.error else message
        return f"Worker did not apply any edit: {self.error}"

    def _render_summary(self, response: str) -> str:
        prose = _prose_summary(response)
        lines = [prose] if prose else []
        if self.changed_files:
            deleted = set(self.deleted_files)
            lines += ["", "Changed files:"]
            lines += [
                f"- {path}" + (" (deleted)" if path in deleted else "")
                for path in self.changed_files
            ]
        else:
            lines += ["", "No file changes were applied."]
        if self._notes:
            lines += ["", "Notes:"] + [f"- {note}" for note in self._notes]
        if self.error:
            lines += ["", f"Errors: {self.error}"]
        return "\n".join(lines).strip()
