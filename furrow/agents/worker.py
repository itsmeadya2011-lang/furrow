from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, WorkerResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Worker agent that executes tasks using the LLM client and file tools."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path(workspace) if workspace else self.client.settings.workspace

    async def run(self) -> dict:
        """Execute the assigned task using the LLM with file tool context.

        Returns:
            dict with keys `files_changed`, `summary`, and `complete`.
        """
        try:
            files = self.client.list_files(self.workspace)
            file_listing = "\n".join(files) if files else "(empty workspace)"
            context = f"Workspace files:\n{file_listing}\n"
            prompt = (
                f"{WORKER_PROMPT}\n\n"
                f"{context}\n"
                f"Task: {self.task.description}\n"
                f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
                "You have access to these tools: list_files, read_file, write_file.\n"
                "Return a JSON object with keys: files_changed, summary, complete."
            )
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        except Exception as exc:
            return WorkerResult(
                files_changed=[],
                summary=f"LLM call failed: {exc}",
                complete=False,
            ).model_dump()

        try:
            parsed = json.loads(response)
            return WorkerResult(
                files_changed=parsed.get("files_changed", []),
                summary=parsed.get("summary", response),
                complete=bool(parsed.get("complete", False)),
            ).model_dump()
        except (json.JSONDecodeError, ValueError):
            return {
                "files_changed": [],
                "summary": response,
                "complete": "complete" in response.lower(),
            }
