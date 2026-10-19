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
        workspace = self.client.settings.workspace
        parts = [WORKER_PROMPT, "", f"Task: {self.task.description}", f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}", ""]

        try:
            available = self.client.list_files(workspace)
            parts.append("Available files in workspace:")
            parts.append("\n".join(available) if available else "(none)")
            parts.append("")
        except Exception:
            pass

        if self.task.files:
            parts.append("Current file contents:")
            for rel in self.task.files:
                path = workspace / rel
                try:
                    content = await self.client.read_file(path)
                    parts.append(f"--- {rel} ---")
                    parts.append(content)
                except Exception:
                    parts.append(f"--- {rel} --- (does not exist; create it)")
                parts.append("")

        prompt = "\n".join(parts)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        pattern = re.compile(r"FILE:\s*(.+?)\n```[a-zA-Z]*\n(.*?)```", re.DOTALL)
        matches = pattern.findall(response)
        changed = []
        for path_str, content in matches:
            path_str = path_str.strip()
            try:
                await self.client.write_file(workspace / path_str, content)
                changed.append(path_str)
            except Exception:
                continue

        if not changed:
            return response

        return "Changed files: " + ", ".join(changed)
