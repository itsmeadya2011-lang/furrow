from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.jsonutils import extract_json
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        raw = await self.client.complete(prompt, model=self.client.settings.worker_model)

        m = re.search(r"```edits\s*(.*?)```", raw, re.DOTALL)
        summary = raw
        errors: list[str] = []

        if m:
            summary = raw[: m.start()] + raw[m.end() :]
            try:
                edits = extract_json(m.group(1))
            except Exception:
                edits = None
            if isinstance(edits, list):
                for item in edits:
                    if not isinstance(item, dict) or "path" not in item or "content" not in item:
                        errors.append(f"Skipped invalid edit entry: {item!r}")
                        continue
                    try:
                        await self.client.write_file(item["path"], item["content"])
                    except Exception as exc:
                        errors.append(f"Failed to write {item['path']}: {exc}")

        summary = summary.strip()
        if errors:
            summary += "\n\nEdit errors:\n- " + "\n- ".join(errors)
        return summary
