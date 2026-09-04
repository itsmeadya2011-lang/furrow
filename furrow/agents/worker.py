from __future__ import annotations

from pathlib import Path

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

log = get_logger(__name__)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or self.client.settings.workspace
        self.settings = self.client.settings

    async def run(self) -> str:
        files = self.task.files or [str(f) for f in self._list_workspace_files()]
        context = await self._gather_context(files)
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(files) if files else 'any'}\n"
            f"\n--- Existing file context ---\n{context}\n"
            f"--- End context ---\n"
            f"\nNow implement the task and return a concise summary."
        )
        log.debug("worker starting task", task_id=self.task.id, files=files)
        return await self.client.complete(
            prompt, system=WORKER_SYSTEM, model=self.settings.worker_model
        )

    def _list_workspace_files(self, max_files: int = 50) -> list[Path]:
        return sorted(self.workspace.rglob("*"))[:max_files] if self.workspace.exists() else []

    async def _gather_context(self, files: list[str]) -> str:
        parts: list[str] = []
        for f in files[:10]:
            path = Path(f)
            if not path.is_absolute():
                path = self.workspace / path
            if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".so"}:
                continue
            try:
                if path.exists() and path.is_file():
                    content = await self.client.read_file(path)
                    parts.append(f"# File: {path}\n```\n{content[:3000]}\n```\n")
            except Exception as e:
                log.debug("could not read file for context", file=str(path), error=str(e))
        return "\n".join(parts)
