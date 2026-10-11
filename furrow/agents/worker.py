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
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        file_contents = {}
        missing_files = []

        for path in self.task.files or []:
            try:
                content = await self.client.read_file(path)
                file_contents[path] = content
            except FileNotFoundError:
                missing_files.append(path)
            except Exception:
                missing_files.append(path)

        prompt_parts = [f"Task: {self.task.description}"]

        if file_contents:
            prompt_parts.append("Current file contents:")
            for path, content in file_contents.items():
                prompt_parts.append(f"--- {path} ---")
                prompt_parts.append(content)
                prompt_parts.append("--- end ---")

        if missing_files:
            prompt_parts.append(f"Note: These files were not found: {', '.join(missing_files)}")

        prompt = "\n".join(prompt_parts)

        response = await self.client.complete(prompt, model=self.client.settings.worker_model, system=WORKER_PROMPT)

        try:
            result = json.loads(response)
            files = result.get("files", {})
            summary = result.get("summary", "Task completed")

            for path, content in files.items():
                try:
                    await self.client.write_file(path, content)
                except Exception:
                    pass

            return summary
        except json.JSONDecodeError:
            return response