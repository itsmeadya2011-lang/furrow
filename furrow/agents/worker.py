from __future__ import annotations

import json
from pathlib import Path

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)

    async def run(self) -> str:
        # Read existing contents of the files the task is meant to touch.
        file_contents: dict[str, str] = {}
        for rel in self.task.files:
            p = Path(rel)
            if p.exists():
                try:
                    file_contents[rel] = await self.client.read_file(p)
                except Exception:
                    file_contents[rel] = "<unreadable>"

        files_list = ", ".join(self.task.files) if self.task.files else "any"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_list}\n"
            f"Existing file contents:\n{json.dumps(file_contents, indent=2)}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse the JSON file-operation response.
        writes: list[dict[str, str]] = []
        summary = response.strip()
        try:
            data = json.loads(response)
            summary = data.get("summary", response).strip()
            writes = data.get("writes", []) or []
        except (json.JSONDecodeError, ValueError):
            # Graceful degradation: if the model didn't return JSON, report the
            # raw response as the summary and make no changes.
            return summary

        # Execute every write inside the workspace, guarding against path
        # traversal so writes never escape the project directory.
        workspace = self.settings.workspace.resolve()
        for write in writes:
            rel = write.get("path")
            content = write.get("content")
            if not rel or content is None:
                continue
            target = (workspace / rel).resolve()
            try:
                target.relative_to(workspace)
            except ValueError:
                continue  # skip writes that escape the workspace
            await self.client.write_file(target, content)

        return summary
