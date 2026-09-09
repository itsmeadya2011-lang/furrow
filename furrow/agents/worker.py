from __future__ import annotations

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
        file_contents = {}
        for file_path in self.task.files or []:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""

        files_context = ""
        if file_contents:
            files_context = "\n\nFiles provided:\n"
            for path, content in file_contents.items():
                files_context += f"\n---FILE: {path}---\n{content}\n---ENDFILE---"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        changes_summary = await self._parse_and_write_files(response)

        return changes_summary

    async def _parse_and_write_files(self, response: str) -> str:
        pattern = r"---FILE:\s*(.+?)---\n(.*?)\n---ENDFILE---"
        matches = re.findall(pattern, response, re.DOTALL)

        written_files = []
        for file_path, content in matches:
            file_path = file_path.strip()
            content = content.rstrip("\n")
            await self.client.write_file(file_path, content)
            written_files.append(file_path)

        if written_files:
            summary = f"Modified {len(written_files)} file(s): {', '.join(written_files)}"
        else:
            summary = "No file changes detected in response"

        return f"{summary}\n\n{response}"