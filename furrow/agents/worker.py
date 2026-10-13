from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_FILE_MARKER = re.compile(r"^\s*FILE:\s*(\S.*?)\s*$")
_SUMMARY_MARKER = re.compile(r"^\s*SUMMARY:\s*(.*?)\s*$")
_FENCE = re.compile(r"^\s*`{3,}")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        contents, read_errors = await self._read_task_files()

        prompt = self._build_prompt(contents, read_errors)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        changes, change_summary = _parse_file_changes(response)
        if not changes:
            raise ValueError(
                f"No file changes could be parsed for task {self.task.id}. "
                f"LLM response: {response[:500]}"
            )

        modified, created, write_errors = await self._write_changes(changes, contents)

        lines = [change_summary or f"Task {self.task.id}: {self.task.description}"]
        if modified:
            lines.append(f"Modified: {', '.join(modified)}")
        if created:
            lines.append(f"Created: {', '.join(created)}")
        if write_errors:
            lines.append(f"Failed to write: {'; '.join(write_errors)}")
        if read_errors:
            detail = "; ".join(f"{path} ({reason})" for path, reason in read_errors.items())
            lines.append(f"Could not read before editing: {detail}")
        return "\n".join(lines)

    async def _read_task_files(self) -> tuple[dict[str, str], dict[str, str]]:
        contents: dict[str, str] = {}
        errors: dict[str, str] = {}
        for path in self.task.files:
            try:
                contents[path] = await self.client.read_file(path)
            except FileNotFoundError:
                errors[path] = "file does not exist yet"
            except Exception as exc:
                errors[path] = str(exc) or exc.__class__.__name__
        return contents, errors

    async def _write_changes(
        self, changes: list[tuple[str, str]], existing: dict[str, str]
    ) -> tuple[list[str], list[str], list[str]]:
        modified: list[str] = []
        created: list[str] = []
        errors: list[str] = []
        for path, content in changes:
            try:
                await self.client.write_file(path, content)
                if path in existing:
                    modified.append(path)
                else:
                    created.append(path)
            except Exception as exc:
                errors.append(f"{path} ({str(exc) or exc.__class__.__name__})")
        return modified, created, errors

    def _build_prompt(self, contents: dict[str, str], read_errors: dict[str, str]) -> str:
        sections = [
            WORKER_PROMPT,
            "",
            f"Task: {self.task.description}",
        ]
        if self.task.files:
            sections.append(f"Files to work on: {', '.join(self.task.files)}")
        else:
            sections.append("Files to work on: any files the task requires")
        sections.extend(["", "Current file contents:"])
        if not self.task.files:
            sections.append("(no files listed; create whatever the task needs)")
        for path in self.task.files:
            sections.append(f"--- {path} ---")
            if path in contents:
                sections.append(contents[path])
            else:
                reason = read_errors.get(path, "file does not exist yet")
                sections.append(f"(not available: {reason})")
            sections.append("")
        sections.append("Implement the task now and return the file changes.")
        return "\n".join(sections)


def _parse_file_changes(response: str) -> tuple[list[tuple[str, str]], str]:
    changes: list[tuple[str, str]] = []
    summary_parts: list[str] = []
    path: str | None = None
    buffer: list[str] = []
    in_summary = False

    def flush() -> None:
        nonlocal path, buffer, in_summary
        if path is not None:
            changes.append((path, _strip_fences(buffer)))
        path = None
        buffer = []
        in_summary = False

    for line in response.splitlines():
        marker = _FILE_MARKER.match(line)
        if marker is not None:
            flush()
            cleaned = _clean_path(marker.group(1))
            if cleaned:
                path = cleaned
            continue
        summary = _SUMMARY_MARKER.match(line)
        if summary is not None:
            flush()
            in_summary = True
            if summary.group(1).strip():
                summary_parts.append(summary.group(1).strip())
            continue
        if in_summary:
            if line.strip():
                summary_parts.append(line.strip())
            continue
        if path is None:
            continue
        buffer.append(line)
    flush()
    return changes, " ".join(summary_parts).strip()


def _clean_path(raw: str) -> str:
    path = raw.strip().strip("`").strip("'\"").strip()
    if path.startswith("./"):
        path = path[2:]
    return path


def _strip_fences(lines: list[str]) -> str:
    content = list(lines)
    while content and not content[0].strip():
        content.pop(0)
    if content and _FENCE.match(content[0]):
        content.pop(0)
    while content and not content[-1].strip():
        content.pop()
    if content and _FENCE.match(content[-1]):
        content.pop()
    while content and not content[-1].strip():
        content.pop()
    return "\n".join(content)
