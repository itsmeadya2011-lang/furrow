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
        # Gather existing contents of the files the task is to touch.
        file_contents: dict[str, str] = {}
        for path in self.task.files:
            try:
                file_contents[path] = await self.client.read_file(path)
            except Exception:
                file_contents[path] = ""  # file does not exist yet

        context = ""
        if file_contents:
            context = "\nExisting file contents:\n" + "\n".join(
                f"--- {path} ---\n{content}" for path, content in file_contents.items()
            )

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{context}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            edits = json.loads(response)
        except json.JSONDecodeError:
            # Fall back: return the raw model output as a summary so the
            # orchestrator can surface what the worker intended.
            return f"Worker produced non-JSON output for task {self.task.id}:\n{response}"

        if not isinstance(edits, dict):
            return f"Worker output for task {self.task.id} was not a JSON object:\n{response}"

        applied: list[str] = []
        for path, content in edits.items():
            if not isinstance(content, str):
                continue
            await self.client.write_file(path, content)
            applied.append(path)

        return f"Task {self.task.id} complete — applied changes to {len(applied)} file(s): {', '.join(applied) if applied else 'none'}"
