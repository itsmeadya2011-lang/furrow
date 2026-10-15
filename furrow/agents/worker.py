from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_FILE_BLOCK_RE = re.compile(r"FILE: (.+?)\n```(?:\w*)\n(.*?)```", re.DOTALL)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path(workspace) if workspace else Path.cwd()

    async def _read_task_files(self) -> dict[str, str]:
        contents: dict[str, str] = {}
        for rel_path in self.task.files:
            try:
                contents[rel_path] = await self.client.read_file(self.workspace / rel_path)
            except FileNotFoundError:
                contents[rel_path] = ""
        return contents

    def _build_prompt(self, file_contents: dict[str, str]) -> str:
        files_section = "\n".join(
            f"{path}:\n```\n{content}\n```"
            for path, content in file_contents.items()
        )
        return WORKER_PROMPT.format(
            task_description=self.task.description,
            files_section=files_section,
        )

    def _parse_changes(self, response: str) -> list[tuple[str, str]]:
        changes: list[tuple[str, str]] = []
        for match in _FILE_BLOCK_RE.finditer(response):
            path = match.group(1).strip()
            content = match.group(2)
            changes.append((path, content))
        return changes

    async def run(self) -> str:
        file_contents = await self._read_task_files()
        prompt = self._build_prompt(file_contents)
        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        changes = self._parse_changes(response)
        for path, content in changes:
            await self.client.write_file(self.workspace / path, content)

        if changes:
            changed = "\n".join(f"- {path}" for path, _ in changes)
            return f"Completed task '{self.task.description}'.\nUpdated files:\n{changed}"
        return f"Completed task '{self.task.description}'. No file changes detected."
