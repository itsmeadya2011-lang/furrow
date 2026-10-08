from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings_obj=settings)

    async def run(self) -> str:
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""
            except Exception as e:  # noqa: BLE001
                file_contents[file_path] = f"Error reading: {e}"

        context = ""
        if file_contents:
            context = "\n\nCurrent file contents:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            operations = json.loads(response)
            if isinstance(operations, dict) and "files" in operations:
                for file_op in operations["files"]:
                    path = file_op.get("path")
                    content = file_op.get("content", "")
                    if path:
                        await self.client.write_file(path, content)
                return operations.get("summary", "Completed file operations")
        except (json.JSONDecodeError, ValueError):
            pass

        return response