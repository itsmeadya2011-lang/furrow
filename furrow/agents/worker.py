from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()

# Regex to extract file-change blocks from the LLM response.
# Format:
#   [[FILE: path/to/file.py]]
#   <file content>
#   [[END]]
_FILE_BLOCK_RE = re.compile(
    r"\[\[FILE:\s*(.+?)\]\]\n(.*?)\n\[\[END\]\]",
    re.DOTALL,
)


class WorkerAgent:
    """Executes a single task, reading context and applying generated file changes."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        task = self.task
        logger.info("worker_run", task_id=task.id, description=task.description)

        # --- Gather file context so the LLM has grounding ---
        file_context = await self._read_file_context(task.files)

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {task.description}\n"
            f"Files to touch: {', '.join(task.files) if task.files else 'any'}\n"
        )
        if file_context:
            prompt += f"\n--- Existing file context ---\n{file_context}\n--- End context ---\n"
        prompt += (
            "\nIf you modify or create files, wrap each file's full content in "
            "blocks like:\n"
            "[[FILE: path/to/file.py]]\n"
            "<full file content here>\n"
            "[[END]]\n"
            "Return your summary after the file blocks.\n"
        )

        result = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # --- Apply file changes extracted from the LLM response ---
        changes = self._extract_file_changes(result)
        for path, content in changes:
            await self._apply_change(path, content)

        summary = self._build_summary(changes, result)
        logger.info(
            "worker_done", task_id=task.id, result_len=len(result), files_written=len(changes)
        )
        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _read_file_context(self, files: list[str]) -> str:
        """Read the files listed in the task so the LLM has context."""
        if not files:
            return ""
        chunks: list[str] = []
        for rel_path in files:
            full_path = Path(rel_path)
            if not full_path.is_absolute():
                workspace = self.client.settings.workspace
                full_path = workspace / rel_path
            try:
                content = await self.client.read_file(full_path)
                chunks.append(f"--- {rel_path} ---\n{content}")
            except (FileNotFoundError, IsADirectoryError, OSError):
                # File doesn't exist yet (e.g. new file) — that's fine.
                chunks.append(f"--- {rel_path} ---\n(file does not exist yet)")
        return "\n".join(chunks)

    def _extract_file_changes(self, response: str) -> list[tuple[str, str]]:
        """Parse ``[[FILE: ...]] ... [[END]]`` blocks from the LLM response."""
        return [
            (match.group(1).strip(), match.group(2).rstrip("\n"))
            for match in _FILE_BLOCK_RE.finditer(response)
        ]

    async def _apply_change(self, path: str, content: str) -> Path:
        """Write ``content`` to ``path`` (creating parent dirs as needed)."""
        workspace = self.client.settings.workspace
        target = Path(path)
        if not target.is_absolute():
            target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        await self.client.write_file(target, content)
        logger.info("file_written", path=str(target))
        return target

    def _build_summary(self, changes: list[tuple[str, str]], raw: str) -> str:
        """Build a human-readable summary of what the worker did."""
        files_changed = [p for p, _ in changes]
        if changes:
            file_list = ", ".join(files_changed)
            summary = f"Modified {len(changes)} file(s): {file_list}.\n\n"
        else:
            summary = "No file changes detected.\n\n"
        # Strip file blocks from the raw response to show only the prose summary.
        prose = _FILE_BLOCK_RE.sub("", raw).strip()
        if prose:
            summary += prose
        return summary
