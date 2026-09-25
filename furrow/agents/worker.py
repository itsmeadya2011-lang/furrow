from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents._util import parse_json_response
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger(__name__)


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_context = await self._existing_files_context()

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{files_context}"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        data = parse_json_response(response)
        if not data or "files" not in data:
            log.warning("worker_non_json_response", task=self.task.id)
            return response.strip()

        for entry in data.get("files", []):
            path = entry.get("path")
            content = entry.get("content", "")
            if path:
                await self.client.write_file(path, content)

        summary = data.get("summary", "").strip()
        return summary or response.strip()

    async def _existing_files_context(self) -> str:
        if not self.task.files:
            return ""
        blocks = []
        for rel in self.task.files:
            path = self.client.settings.workspace / rel
            if path.exists():
                content = await self.client.read_file(rel)
                blocks.append(f"--- {rel} ---\n{content}")
        if not blocks:
            return ""
        return "\nExisting files:\n" + "\n\n".join(blocks) + "\n"
