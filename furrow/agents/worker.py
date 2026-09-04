from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)

_MAX_FILE_CHARS = 8000


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def _get_workspace_context(self, workspace: Path, files: list[str]) -> str:
        all_files = self.client.list_files(workspace)
        lines: list[str] = ["Project structure:"]
        for f in all_files:
            lines.append(f"  {f}")

        if files:
            lines.append("")
            lines.append("Relevant file contents:")
            for file_path in files:
                full_path = workspace / file_path
                if not full_path.is_file():
                    lines.append(f"--- {file_path} (not found) ---")
                    continue
                try:
                    content = await self.client.read_file(full_path)
                except Exception as e:
                    lines.append(f"--- {file_path} (read error: {e}) ---")
                    continue

                if len(content) > _MAX_FILE_CHARS:
                    content = content[:_MAX_FILE_CHARS] + f"\n... [truncated, total {len(content)} chars]"
                lines.append(f"--- {file_path} ---")
                lines.append(content)

        return "\n".join(lines)

    async def run(self, workspace: Path | None = None) -> str:
        workspace = workspace or Path.cwd()
        context = await self._get_workspace_context(workspace, self.task.files)
        log.debug("worker.context", task_id=self.task.id, context=context[:2000])
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"Workspace context:\n{context}\n"
        )
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
