from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, TaskResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            result = TaskResult(**data)
        except (json.JSONDecodeError, ValueError, TypeError):
            # If the LLM didn't return valid JSON, treat the raw response as the summary
            return response.strip()

        for edit in result.edits:
            try:
                if edit.operation == "create":
                    await self.client.write_file(edit.file, edit.content or "")
                elif edit.operation == "edit":
                    await self.client.edit_file(edit.file, edit.old, edit.new)
                elif edit.operation == "delete":
                    await self.client.delete_file(edit.file)
            except Exception as e:
                result.issues.append(f"Failed to {edit.operation} {edit.file}: {e}")

        summary = result.summary or "No summary"
        if result.issues:
            summary += "\nIssues:\n" + "\n".join(f"  - {i}" for i in result.issues)
        return summary
