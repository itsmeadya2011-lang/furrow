from __future__ import annotations

import json
from pathlib import Path
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

    def _resolve_path(self, path: str) -> Path:
        workspace = Path(self.client.settings.workspace)
        target = (workspace / path).resolve()
        try:
            target.relative_to(workspace.resolve())
        except ValueError:
            target = (workspace / Path(path).name).resolve()
        return target

    async def run(self) -> str:
        files: list[str] = list(self.task.files)

        existing_lines: list[str] = []
        for path in files:
            try:
                content = await self.client.read_file(path)
                existing_lines.append(f"Existing content of {path}:\n{content}")
            except FileNotFoundError:
                existing_lines.append(f"Existing content of {path}:\n[new file]")

        prompt = (
            f"{WORKER_PROMPT}\n"
            f"Goal context:\nGoal: {self.task.description}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(files) if files else 'any'}\n"
        )
        if existing_lines:
            prompt += "\n" + "\n\n".join(existing_lines) + "\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        text = response.strip()
        if text.startswith("```"):
            text = text.lstrip("`")
            text = text.removeprefix("json").strip()
            text = text.rstrip("`").strip()
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return (
                "WARNING: could not parse worker JSON output; raw response below:\n" + response
            )

        files_out: list[dict] = data.get("files", []) if isinstance(data, dict) else []
        summary: str = data.get("summary", "") if isinstance(data, dict) else ""

        if files_out:
            for entry in files_out:
                file_path = entry.get("path")
                content = entry.get("content", "")
                if file_path is None:
                    continue
                resolved = self._resolve_path(file_path)
                await self.client.write_file(resolved, content)

        return summary
