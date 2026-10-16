from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_OUTPUT_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Read existing files so the LLM can modify them in place.
        file_contents: dict[str, str] = {}
        for path in self.task.files:
            try:
                file_contents[path] = await self.client.read_file(path)
            except Exception:
                file_contents[path] = "(file does not exist or could not be read)"

        files_str = "\n".join(
            f"--- {path} ---\n{content}"
            for path, content in file_contents.items()
        ) if file_contents else "No existing files to read."

        prompt = (
            f"{WORKER_OUTPUT_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Existing file contents:\n{files_str}\n"
        )

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        try:
            data = json.loads(response)
            files = data.get("files", [])
            summary = data.get("summary", "No summary provided.")

            for file_info in files:
                path = file_info["path"]
                content = file_info["content"]
                await self.client.write_file(path, content)

            return summary
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            # Fallback: treat the raw response as a summary only.
            return response.strip()
