from __future__ import annotations

import json

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
        files = self.task.files
        contents: dict[str, str] = {}
        if files:
            for f in files:
                try:
                    contents[f] = await self.client.read_file(f)
                except (FileNotFoundError, OSError, IOError):
                    contents[f] = ""
        else:
            try:
                listed = self.client.list_files(".")
                files = listed
                for f in listed:
                    try:
                        contents[f] = await self.client.read_file(f)
                    except (FileNotFoundError, OSError, IOError):
                        contents[f] = ""
            except (FileNotFoundError, OSError, IOError):
                files = []
                contents = {}

        parts = [WORKER_PROMPT]
        parts.append(f"Task: {self.task.description}")
        parts.append(f"Files to touch: {', '.join(files) if files else 'any'}")
        if contents:
            parts.append("Current file contents:")
            for path, content in contents.items():
                parts.append(f"--- {path} ---\n{content}")
        prompt = "\n\n".join(parts)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
            for edit in data.get("edits", []):
                await self.client.write_file(edit["path"], edit["content"])
            return data.get("summary", "Task completed.")
        except (json.JSONDecodeError, KeyError, TypeError):
            return response.strip() or "Task completed."
