from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        context = await self._gather_context()
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {files_str}\n\nWorkspace context:\n{context}\n"
        log.info("Worker starting task %s", self.task.id)
        return await self.client.complete(prompt, model=self.client.settings.worker_model)

    async def _gather_context(self) -> str:
        """Collect workspace context to give the worker awareness of existing code."""
        workspace = self.client.settings.workspace
        files = self.client.list_files(workspace)
        # Limit context to a reasonable number of files
        files = files[:100]
        if not files:
            return "No existing files found."

        # Focus on files relevant to the task when hints are available.
        relevant = self.task.files or files
        snippets = []
        for f in relevant[:20]:
            full_path = workspace / f
            if not full_path.exists():
                continue
            try:
                content = await self.client.read_file(full_path)
                # Truncate to keep prompt manageable
                snippet = content[:2000]
                if len(content) > 2000:
                    snippet += "\n... (truncated)"
                snippets.append(f"--- {f} ---\n{snippet}")
            except OSError as e:
                log.debug("Could not read %s: %s", f, e)

        if not snippets:
            return "Workspace files:\n" + "\n".join(files[:50])
        return "\n\n".join(snippets)
