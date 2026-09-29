from __future__ import annotations

import json
from pathlib import Path

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel, WorkerResult
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> WorkerResult:
        # Gather existing file contents so the LLM has context.
        file_context: list[str] = []
        for rel_path in self.task.files:
            abs_path = Path(self.client.settings.workspace) / rel_path
            if abs_path.exists():
                content = await self.client.read_file(abs_path)
                file_context.append(f"--- {rel_path} (existing) ---\n{content}")
            else:
                file_context.append(f"--- {rel_path} (new file) ---")

        files_str = "\n\n".join(file_context) if file_context else "(no files specified)"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n\n"
            f"Files to touch:\n{files_str}\n"
        )

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        try:
            data = json.loads(response)
            result = WorkerResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            result = WorkerResult(
                summary=f"Failed to parse worker output: {e}\nResponse: {response}",
                files=[],
            )

        # Write all files to the workspace.
        for file_edit in result.files:
            abs_path = Path(self.client.settings.workspace) / file_edit.path
            await self.client.write_file(abs_path, file_edit.content)

        return result
