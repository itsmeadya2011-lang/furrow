from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

console = Console()


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        file_listing = "\n".join(self.client.list_files(self.client.settings.workspace)) or "(none)"

        existing: list[str] = []
        for file in self.task.files:
            try:
                content = await self.client.read_file(self._resolve(file))
            except (OSError, FileNotFoundError):
                continue
            existing.append(f"--- {file} ---\n{content}")
        file_contents = "\n\n".join(existing) or "(none)"

        target_files = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Target files: {target_files}

Existing files in the workspace:
{file_listing}

Current contents of target files:
{file_contents}

Return a JSON object describing the edits to make, with this exact shape:
{{
  "edits": [
    {{"path": "src/foo.py", "action": "create|modify", "content": "the full new file content"}}
  ],
  "summary": "what was changed"
}}"""

        try:
            response = await self.client.complete_json(prompt, model=self.client.settings.worker_model)
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return await self.client.complete(prompt, model=self.client.settings.worker_model)

        edits = data.get("edits", []) if isinstance(data, dict) else []
        applied = 0
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            path = edit.get("path")
            content = edit.get("content")
            if not path or content is None:
                continue
            await self.client.write_file(self._resolve(path), content)
            applied += 1

        summary = data.get("summary") if isinstance(data, dict) else None
        summary = summary or f"Applied {applied} edit(s)"
        console.print(f"[green]Task {self.task.id}: applied {applied} edit(s)[/green]")
        return summary

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.client.settings.workspace / p
