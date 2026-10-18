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
        files = self.task.files or []
        current_contents: dict[str, str] = {}
        for path in files:
            try:
                current_contents[path] = await self.client.read_file(path)
            except (FileNotFoundError, OSError):
                pass

        context_parts = []
        for path in files:
            if path in current_contents:
                context_parts.append(f"--- {path} ---\n{current_contents[path]}")
            else:
                context_parts.append(f"--- {path} --- (file does not exist)")

        files_list = ", ".join(files) if files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_list}\n\n"
            f"Current file contents:\n" + "\n\n".join(context_parts) + "\n\n"
            "Return the complete new contents for each changed or newly created file as JSON with shape: "
            '{"files": {"<path>": "<full new content>"}, "summary": "<brief summary of changes>"}. '
            "Only include files that actually changed or are newly created."
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return response

        files_to_write = data.get("files", {}) if isinstance(data, dict) else {}
        written: list[str] = []
        for path, content in files_to_write.items():
            await self.client.write_file(path, content)
            written.append(path)

        summary = data.get("summary") if isinstance(data, dict) else None
        if summary:
            return summary
        if written:
            return f"Wrote files: {', '.join(written)}"
        return response
