from __future__ import annotations

import difflib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_FILE_CHARS = 20_000
MAX_PROMPT_CHARS = 60_000
MAX_TREE_ENTRIES = 120
MAX_NOTES_CHARS = 600
MAX_PROBLEMS = 10

_FILE_OPEN = "<<<FILE:"
_EDIT_OPEN = "<<<EDIT:"
_FILE_CLOSE = "<<<END FILE>>>"
_EDIT_CLOSE = "<<<END EDIT>>>"
_SEARCH = "<<<SEARCH>>>"
_REPLACE = "<<<REPLACE>>>"

_BLOCK_RE = re.compile(
    rf"{re.escape(_FILE_OPEN)}[^\n]*>>>.*?{re.escape(_FILE_CLOSE)}"
    rf"|{re.escape(_EDIT_OPEN)}[^\n]*>>>.*?{re.escape(_EDIT_CLOSE)}",
    re.DOTALL,
)


def _iter_blocks(text: str) -> Iterator[tuple[str, str, str]]:
    """Yield ``(kind, raw_path, body)`` for each edit-protocol block, in order.

    ``kind`` is ``"file"`` or ``"edit"``. Closing markers are matched leniently:
    an unterminated block runs to the end of the reply, because models sometimes
    drop the final marker rather than losing the whole change.
    """
    cursor = 0
    while cursor < len(text):
        file_at = text.find(_FILE_OPEN, cursor)
        edit_at = text.find(_EDIT_OPEN, cursor)
        starts = [(at, kind) for at, kind in ((file_at, "file"), (edit_at, "edit")) if at != -1]
        if not starts:
            return
        at, kind = min(starts, key=lambda item: item[0])
        header_end = text.find(">>>", at)
        if header_end == -1:
            return
        raw_path = text[at + len(_FILE_OPEN if kind == "file" else _EDIT_OPEN) : header_end].strip()
        body_start = header_end + 3
        closer = _FILE_CLOSE if kind == "file" else _EDIT_CLOSE
        close_at = text.find(closer, body_start)
        end = len(text) if close_at == -1 else close_at
        yield kind, raw_path, _trim_block(kind, text[body_start:end])
        cursor = len(text) if close_at == -1 else close_at + len(closer)


def _trim_block(kind: str, body: str) -> str:
    """Drop the newline right after the header; keep a file body's final newline."""
    trimmed = body[1:] if body.startswith("\n") else body
    return trimmed.strip("\n") if kind == "edit" else trimmed


def _split_edit(body: str) -> tuple[str, str] | None:
    """Split an EDIT payload into its search and replace halves."""
    search_at = body.find(_SEARCH)
    replace_at = body.find(_REPLACE)
    if search_at == -1 or replace_at == -1 or replace_at < search_at:
        return None
    search = body[search_at + len(_SEARCH) : replace_at].strip("\n")
    replace = body[replace_at + len(_REPLACE) :].strip("\n")
    return search, replace


def _match_window(lines: list[str], needle: list[str], strip_left: bool) -> int:
    """Return the start index of the unique window matching ``needle``, else -1."""
    if not needle or len(needle) > len(lines):
        return -1

    def key(line: str) -> str:
        return line.strip() if strip_left else line.rstrip()

    target = [key(line) for line in needle]
    found = -1
    for start in range(len(lines) - len(needle) + 1):
        if all(key(line) == want for line, want in zip(lines[start:], target, strict=False)):
            if found != -1:
                return -1
            found = start
    return found


def apply_search_replace(existing: str, search: str, replace: str) -> tuple[str | None, str]:
    """Apply a search/replace edit, or explain why it was refused.

    Exact single match wins; otherwise a whitespace-tolerant line match is tried.
    Ambiguity and absence are both reported rather than guessed at.
    """
    if not search.strip():
        return None, "SEARCH text is empty"
    count = existing.count(search)
    if count == 1:
        return existing.replace(search, replace, 1), ""
    if count > 1:
        return None, f"SEARCH text is not unique ({count} occurrences)"
    existing_lines = existing.splitlines()
    search_lines = search.splitlines()
    for strip_left in (False, True):
        start = _match_window(existing_lines, search_lines, strip_left)
        if start == -1:
            continue
        stop = start + len(search_lines)
        merged = [*existing_lines[:start], *replace.splitlines(), *existing_lines[stop:]]
        return "\n".join(merged), ""
    return None, "SEARCH text not found in file"


def _line_delta(before: str, after: str) -> tuple[int, int]:
    """Return (lines_added, lines_removed) between two file versions."""
    if before == after:
        return 0, 0
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    added = 0
    removed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


def _extract_notes(text: str) -> str:
    """Return the prose the model wrote around its blocks, if any."""
    remainder = _BLOCK_RE.sub(" ", text).replace("```", " ")
    for marker in ("<<<", ">>>"):
        head = remainder.find(marker)
        if head != -1:
            remainder = remainder[:head]
    collapsed = " ".join(remainder.split())
    if len(collapsed) > MAX_NOTES_CHARS:
        collapsed = collapsed[:MAX_NOTES_CHARS].rstrip() + "..."
    return collapsed


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = self._workspace()
        existing, context = await self._read_context(workspace)
        prompt = self._build_prompt(existing, context, self._workspace_tree(workspace))
        response = await self.client.complete(
            prompt,
            system=WORKER_PROMPT,
            model=getattr(self.client.settings, "worker_model", None),
        )
        actions, problems = await self._apply(response, workspace)
        summary = "; ".join(actions) if actions else "no file changes"
        notes = _extract_notes(response)
        if notes:
            summary = f"{summary} | notes: {notes}"
        if problems:
            summary = f"{summary} | problems: {'; '.join(problems)}"
        return summary

    def _workspace(self) -> Path:
        raw = getattr(self.client.settings, "workspace", None) or Path.cwd()
        path = Path(raw).expanduser()
        try:
            return path.resolve()
        except OSError:
            return path.absolute()

    def _safe_path(self, workspace: Path, raw: str) -> Path | None:
        """Resolve a model-supplied path inside the workspace, or None if unsafe.

        This is the only place where a path coming from LLM output becomes a real
        filesystem path. ``Path.resolve()`` collapses "..", makes absolute paths
        absolute and follows symlinks, and the result must stay inside the
        workspace, so nothing outside it can ever be written. The workspace root
        itself and directories are refused as well.
        """
        candidate = raw.strip().strip("`'\"")
        if not candidate or candidate.startswith("~"):
            return None
        path = Path(candidate)
        target = path if path.is_absolute() else workspace / path
        try:
            resolved = target.resolve()
        except OSError:
            return None
        if not resolved.is_relative_to(workspace) or resolved == workspace:
            return None
        if resolved.is_dir():
            return None
        return resolved

    async def _read_context(self, workspace: Path) -> tuple[list[str], str]:
        """Read the task's existing files, within a per-file and total budget."""
        existing: list[str] = []
        sections: list[str] = []
        budget = MAX_PROMPT_CHARS
        for raw in self.task.files or []:
            if budget <= 0:
                break
            target = self._safe_path(workspace, raw)
            if target is None or not target.is_file():
                continue
            try:
                text = await self.client.read_file(target)
            except (OSError, ValueError):
                continue
            truncated = text[:MAX_FILE_CHARS]
            marker = "\n... [truncated]" if len(text) > MAX_FILE_CHARS else ""
            section = f"--- existing file: {target.relative_to(workspace).as_posix()}{marker} ---\n"
            section = (section + truncated + "\n")[:budget]
            budget -= len(section)
            existing.append(target.relative_to(workspace).as_posix())
            sections.append(section)
        return existing, "\n".join(sections)

    def _workspace_tree(self, workspace: Path) -> list[str]:
        try:
            names = self.client.list_files(workspace)
        except (OSError, ValueError):
            return []
        visible = [name for name in names if not _hidden(name)]
        return sorted(visible)[:MAX_TREE_ENTRIES]

    def _build_prompt(self, existing: list[str], context: str, tree: list[str]) -> str:
        declared = ", ".join(self.task.files) if self.task.files else "(not specified)"
        if not self.task.files:
            state = "No file was named for this task, so choose the paths yourself."
        elif existing:
            state = f"Existing files you may edit: {', '.join(existing)}. Their contents follow."
        else:
            state = "None of the task files exist yet; treat them as new files."
        blocks = [
            f"Task id: {self.task.id}",
            f"Task: {self.task.description}",
            f"Task files: {declared}",
            state,
        ]
        if context:
            blocks.append(context)
        if tree:
            blocks.append("Workspace layout (relative paths):\n" + "\n".join(tree))
        blocks.append(
            "Reply using the edit protocol: one <<<FILE: path>>> block per created or\n"
            "fully rewritten file, or one <<<EDIT: path>>> block per targeted change, then\n"
            "at most three lines of plain-text notes. Return the smallest complete change."
        )
        return "\n\n".join(blocks)

    async def _apply(self, response: str, workspace: Path) -> tuple[list[str], list[str]]:
        """Apply every block in the response, collecting errors instead of raising."""
        actions: list[str] = []
        problems: list[str] = []
        for kind, raw_path, body in _iter_blocks(response):
            label = raw_path.strip() or "(missing path)"
            target = self._safe_path(workspace, raw_path)
            if target is None:
                problems.append(f"refused {label!r}: path is outside the workspace")
            elif kind == "file":
                problems.extend(await self._apply_file(target, workspace, body, actions))
            else:
                problems.extend(await self._apply_edit(target, workspace, label, body, actions))
            if len(problems) >= MAX_PROBLEMS:
                problems.append("further problems suppressed")
                break
        return actions, problems

    async def _apply_file(
        self,
        target: Path,
        workspace: Path,
        body: str,
        actions: list[str],
    ) -> list[str]:
        rel = target.relative_to(workspace).as_posix()
        before = await self._read_existing(target) or ""
        error = await self._write(target, body)
        if error is not None:
            return [f"failed to write {rel}: {error}"]
        added, removed = _line_delta(before, body)
        actions.append(f"wrote {rel} (+{added}/-{removed} lines)")
        return []

    async def _apply_edit(
        self,
        target: Path | None,
        workspace: Path,
        label: str,
        body: str,
        actions: list[str],
    ) -> list[str]:
        rel = label if target is None else target.relative_to(workspace).as_posix()
        pair = _split_edit(body)
        if pair is None:
            return [f"skipped edit to {rel}: EDIT block is missing SEARCH or REPLACE"]
        search, replace = pair
        if target is None:
            return [f"skipped edit to {rel}: path is outside the workspace"]
        before = await self._read_existing(target)
        if before is None:
            return [f"skipped edit to {rel}: file does not exist, use a FILE block"]
        new_text, reason = apply_search_replace(before, search, replace)
        if new_text is None:
            return [f"skipped edit to {rel}: {reason}"]
        error = await self._write(target, new_text)
        if error is not None:
            return [f"failed to write {rel}: {error}"]
        added, removed = _line_delta(before, new_text)
        actions.append(f"edited {rel} (+{added}/-{removed} lines)")
        return []

    async def _read_existing(self, target: Path | None) -> str | None:
        if target is None:
            return None
        try:
            return await self.client.read_file(target)
        except (OSError, ValueError):
            return None

    async def _write(self, target: Path, content: str) -> str | None:
        try:
            await self.client.write_file(target, content)
        except (OSError, ValueError) as exc:
            return str(exc)
        return None


def _hidden(name: str) -> bool:
    return any(part.startswith(".") for part in Path(name).parts if part not in (".", ".."))
