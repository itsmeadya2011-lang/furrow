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

    async def run(self) -> str:
        workspace = self.client.settings.workspace

        # Read task files for context.
        file_context = ""
        files_read: dict[str, str] = {}
        for f in self.task.files:
            path = Path(f)
            if not path.is_absolute():
                path = workspace / path
            try:
                content = await self.client.read_file(path)
                files_read[f] = content
            except Exception:
                files_read[f] = "<file not found>"

        if files_read:
            file_context = "\n\n--- File Contents ---\n"
            for fname, content in files_read.items():
                file_context += f"\n[{fname}]\n{content}\n"

        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n"
            f"Task: {self.task.description}\n"
            f"Files to work with: {files_str}\n"
            f"{file_context}\n"
        )

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        # Parse structured output and apply changes.
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            for change in changes:
                file_path = change["file"]
                content = change["content"]
                p = Path(file_path)
                if not p.is_absolute():
                    p = workspace / p
                await self.client.write_file(p, content)
            summary = data.get("summary", "Changes applied.")
            return summary
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            # Fallback: return the raw LLM response as a summary.
            return response
