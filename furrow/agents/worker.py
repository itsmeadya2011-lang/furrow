from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

FILE_EDIT_PATTERN = re.compile(
    r"""^FILE:\s*(?P<path>\S+)\n(?P<fence>```)(?P<lang>\w+)?\n(?P<content>.*?)(?P=fence)""",
    re.DOTALL | re.MULTILINE,
)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        for filepath in self.task.files:
            try:
                content = await self.client.read_file(filepath)
            except FileNotFoundError:
                content = "<file does not exist yet>"
            prompt += f"\n--- Existing contents of {filepath} ---\n{content}\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        written_files: list[str] = []
        matches = FILE_EDIT_PATTERN.finditer(response)
        for match in matches:
            path = match.group("path")
            content = match.group("content")
            await self.client.write_file(path, content)
            written_files.append(path)

        if not written_files:
            return f"Task '{self.task.description}' completed. No files written.\nResponse: {response}"

        summary = f"Task '{self.task.description}' completed. Files written:\n"
        for f in written_files:
            summary += f"  • {f}\n"
        return summary
