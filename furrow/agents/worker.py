from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import EDIT_WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def _gather_context(self) -> str:
        """Read task files and list workspace to build LLM context."""
        sections = []

        files_in_workspace = self.client.list_files(self.client.settings.workspace)
        sections.append(f"Workspace files:\n" + "\n".join(files_in_workspace[:50]))

        if self.task.files:
            sections.append("\n# File contents:\n")
            for file_path in self.task.files:
                try:
                    content = await self.client.read_file(file_path)
                    sections.append(f"--- {file_path} ---\n{content}")
                except Exception as e:
                    sections.append(f"--- {file_path} ---\n[Could not read: {e}]")
        else:
            sections.append("\n# No specific files specified; explore the workspace.")

        return "\n".join(sections)

    @staticmethod
    def _parse_edits(response: str) -> tuple[list[dict], str]:
        """Parse LLM response into edits list and summary. Falls back gracefully."""
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "edits" in data:
                edits = data.get("edits", [])
                summary = data.get("summary", "Completed task edits.")
                return edits, summary
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: no edits, return raw response as summary
        return [], response[:500]

    async def run(self) -> str:
        context = await self._gather_context()
        prompt = (
            f"{EDIT_WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"{context}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        edits, summary = self._parse_edits(response)

        for edit in edits:
            file_path = edit.get("file")
            content = edit.get("content", "")
            if file_path:
                await self.client.write_file(file_path, content)

        return summary
