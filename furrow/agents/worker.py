from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.utils import clean_json_response
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        existing_contents = {}
        for filepath in self.task.files:
            try:
                existing_contents[filepath] = await self.client.read_file(filepath)
            except Exception:
                existing_contents[filepath] = None

        files_context = ""
        for filepath, content in existing_contents.items():
            if content is not None:
                files_context += f"\n\n--- {filepath} (current contents) ---\n{content}"
            else:
                files_context += f"\n--- {filepath} (does not exist yet) ---"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        changes, summary = self._parse_response(response)
        written_files = []
        for filepath, content in changes.items():
            await self.client.write_file(filepath, content)
            written_files.append(filepath)

        if written_files:
            return f"Wrote {len(written_files)} file(s): {', '.join(written_files)}. {summary or response}"
        return summary or response

    def _parse_response(self, response: str) -> tuple[dict[str, str], str]:
        cleaned = clean_json_response(response)
        try:
            data = json.loads(cleaned)
            changes = data.get("changes", {})
            if isinstance(changes, dict):
                return changes, data.get("summary", "")
        except (json.JSONDecodeError, ValueError):
            pass
        return {}, response
