from __future__ import annotations

import json
import re
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
        file_contents: dict[str, str] = {}
        for path in self.task.files:
            try:
                file_contents[path] = await self.client.read_file(path)
            except FileNotFoundError:
                file_contents[path] = ""

        files_section = ""
        if file_contents:
            files_section = "\nCurrent file contents:\n" + "\n".join(
                f"### {path}\n```\n{content}\n```" for path, content in file_contents.items()
            )

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}"
            f"{files_section}\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        data = self._extract_json(response)
        files = data.get("files", []) if isinstance(data, dict) else []

        for file_change in files:
            await self.client.write_file(file_change["path"], file_change["content"])

        return f"Task {self.task.id} completed: wrote {len(files)} file(s)."

    @staticmethod
    def _extract_json(response: str) -> dict:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        text = match.group(1) if match else response
        return json.loads(text)
