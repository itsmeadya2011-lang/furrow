from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

# Bounds on the repository context we are willing to put in front of the model.
# An unbounded file dump would blow the context window and cost money per cycle.
MAX_FILE_CHARS = 6000
MAX_PROMPT_CHARS = 20000
MAX_LISTED_FILES = 200
# An LLM that wants to rewrite 40 files in one turn is a failure mode, not a plan.
MAX_EDITS = 10

TRUNCATION_NOTE = "\n... [truncated: showing first {shown} of {total} characters] ..."


class WorkerAgent:
    """Executes a single task by actually writing files to the workspace.

    The model is shown the relevant repository context and must answer with a
    JSON envelope of full-file contents (``WORKER_PROMPT``); every accepted edit
    is written to disk before :meth:`run` returns.
    """

    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.settings: Settings = settings or (
            client.settings if client is not None else default_settings
        )
        self.client = client or LLMClient(settings=self.settings)

    # ------------------------------------------------------------------ path
    def resolve_workspace_path(self, raw_path: str) -> Path | None:
        """Resolve ``raw_path`` against the workspace, or ``None`` if it escapes.

        Security boundary: the model output is untrusted, so nothing is written
        unless the fully resolved path is contained by the resolved workspace.
        This rejects absolute paths outside the workspace, ``..`` traversal and
        symlink escapes (the target of every path component is resolved).
        """
        candidate = raw_path.strip()
        if not candidate:
            return None
        workspace = Path(self.settings.workspace).resolve()
        try:
            resolved = Path(workspace, candidate).resolve()
        except (OSError, RuntimeError, ValueError):
            logger.warning("Rejecting edit with unusable path: %r", raw_path)
            return None
        if resolved != workspace and workspace not in resolved.parents:
            logger.warning("Rejecting edit outside workspace: %r -> %s", raw_path, resolved)
            return None
        return resolved

    # -------------------------------------------------------------- prompting
    async def build_prompt(self) -> str:
        """Assemble the instruction, the task, and bounded repository context."""
        sections: list[str] = [
            WORKER_PROMPT,
            f"Workspace root: {Path(self.settings.workspace).resolve()}",
            f"Task id: {self.task.id}",
            f"Task: {self.task.description}",
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}",
        ]
        sections.extend(await self._file_context())
        sections.append(self._listing_context())
        return _cap("\n\n".join(s for s in sections if s), MAX_PROMPT_CHARS)

    async def _file_context(self) -> list[str]:
        blocks: list[str] = []
        for raw_path in self.task.files:
            if self.resolve_workspace_path(raw_path) is None:
                blocks.append(f"### {raw_path}\n(not readable: outside workspace)")
                continue
            try:
                content = await self.client.read_file(Path(self.settings.workspace) / raw_path)
            except (OSError, ValueError) as exc:
                blocks.append(f"### {raw_path}\n(new file or unreadable: {exc})")
                continue
            blocks.append(f"### {raw_path}\n```\n{_cap(content, MAX_FILE_CHARS)}\n```")
        return blocks

    def _listing_context(self) -> str:
        try:
            files = self.client.list_files(self.settings.workspace, limit=MAX_LISTED_FILES)
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            logger.debug("Could not list workspace", exc_info=True)
            return f"Project listing unavailable: {exc}"
        if not files:
            return ""
        listing = "\n".join(f"- {name}" for name in files)
        note = "\n... [listing truncated] ..." if len(files) >= MAX_LISTED_FILES else ""
        return f"Project files:\n{listing}{note}"

    # --------------------------------------------------------------- response
    def parse_response(self, response: str) -> dict[str, Any]:
        """Return the JSON envelope, or a summary-only fallback.

        A response that is not JSON at all is not fatal: it is reported as a
        summary with no edits so the cycle continues instead of crashing.
        """
        try:
            data = json.loads(extract_json(response))
        except (ValueError, TypeError) as exc:
            logger.warning("Worker response was not JSON, treating as summary: %s", exc)
            return {"summary": response.strip(), "edits": [], "issues": [f"unparsable response: {exc}"]}
        if not isinstance(data, dict):
            return {"summary": response.strip(), "edits": [], "issues": ["response JSON was not an object"]}
        return data

    async def apply_edits(self, edits: Any) -> tuple[list[str], list[str]]:
        """Write every valid, contained edit. Returns ``(written, problems)``."""
        written: list[str] = []
        problems: list[str] = []
        if not isinstance(edits, list):
            return written, ["no 'edits' list in response"]
        if len(edits) > MAX_EDITS:
            problems.append(f"ignored {len(edits) - MAX_EDITS} edit(s) above the {MAX_EDITS} limit")
            edits = edits[:MAX_EDITS]
        for edit in edits:
            if not isinstance(edit, dict):
                problems.append(f"skipped malformed edit: {edit!r}")
                continue
            path = edit.get("path")
            content = edit.get("content")
            if not isinstance(path, str) or not path.strip():
                problems.append(f"skipped edit without a path: {edit!r}")
                continue
            if not isinstance(content, str):
                problems.append(f"skipped edit to {path}: content is not a string")
                continue
            resolved = self.resolve_workspace_path(path)
            if resolved is None:
                problems.append(f"rejected edit to {path}: outside workspace")
                continue
            try:
                await self.client.write_file(resolved, content)
            except (OSError, ValueError) as exc:
                problems.append(f"failed writing {path}: {exc}")
                logger.warning("Worker failed to write %s", resolved, exc_info=True)
                continue
            written.append(path.strip())
        return written, problems

    # ------------------------------------------------------------------- main
    async def run(self) -> str:
        prompt = await self.build_prompt()
        response = await self.client.complete(prompt, model=self.settings.worker_model)
        data = self.parse_response(response)
        written, problems = await self.apply_edits(data.get("edits"))

        issues = data.get("issues")
        notes = [str(i) for i in issues] if isinstance(issues, list) else []
        summary = data.get("summary")
        summary = str(summary).strip() if summary else ""

        parts = [f"task {self.task.id}: wrote {len(written)} file(s)"]
        if written:
            parts.append(", ".join(written))
        if summary:
            parts.append(f"- {summary}")
        if notes:
            parts.append("issues: " + "; ".join(notes))
        if problems:
            parts.append("problems: " + "; ".join(problems))
        if written or (not notes and not problems):
            return " | ".join(parts)
        raise RuntimeError(f"worker produced no usable edits: {'; '.join(notes + problems)}")


def _cap(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, noting the elision inline."""
    if len(text) <= limit:
        return text
    shown = max(0, limit - 80)
    return text[:shown] + TRUNCATION_NOTE.format(shown=shown, total=len(text))
