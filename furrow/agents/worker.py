from __future__ import annotations

import json

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None) -> None:
        self.task = task
        self.client = client or LLMClient()

    async def run(self) -> str:
        files = self.task.files or []
        contents: dict[str, str] = {}
        for fp in files:
            try:
                contents[fp] = await self.client.read_file(fp)
            except FileNotFoundError:
                contents[fp] = ""

        file_blocks = []
        for fp in files:
            file_blocks.append(f"--- {fp} ---\n{contents.get(fp, '')}")
        files_str = "\n".join(files) if files else "any"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n\n"
            f"Current file contents:\n" + "\n".join(file_blocks) + "\n"
        )

        raw = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

        changes = data.get("changes", {}) if isinstance(data, dict) else {}
        for fp, content in changes.items():
            await self.client.write_file(fp, content)

        summary = data.get("summary") if isinstance(data, dict) else None
        if summary:
            return summary
        written = ", ".join(changes.keys())
        return f"wrote {written}" if written else raw
