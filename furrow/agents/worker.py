from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        context = await self._read_context()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Existing file contents:\n{context}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning("Worker returned non-JSON response: %s", response)
            return f"Worker failed to parse response: {e}\nRaw response: {response}"

        operations = data.get("operations", [])
        summary = data.get("summary", "No summary provided.")

        for op in operations:
            op_type = op.get("type")
            path = op.get("path")
            if op_type == "write" and path and "content" in op:
                await self.client.write_file(path, op["content"])
                logger.info("Wrote %s", path)
            else:
                logger.warning("Skipping unsupported operation: %s", op)

        return summary

    async def _read_context(self) -> str:
        """Read the contents of files listed in the task, for LLM context."""
        parts: list[str] = []
        for path in self.task.files:
            try:
                content = await self.client.read_file(path)
                parts.append(f"=== {path} ===\n{content}\n")
            except Exception as e:  # noqa: BLE001
                parts.append(f"=== {path} ===\n(could not read: {e})\n")
        return "\n".join(parts) if parts else "(no existing files to read)"
