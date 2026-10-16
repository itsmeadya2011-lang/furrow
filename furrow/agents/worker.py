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
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        existing_files: dict[str, str] = {}
        for rel_path in self.task.files:
            p = Path(rel_path)
            if p.exists():
                try:
                    existing_files[rel_path] = await self.client.read_file(p)
                except Exception:
                    existing_files[rel_path] = ""

        existing_files_str = "\n\n".join(
            f"--- {path} ---\n{content}" for path, content in existing_files.items()
        ) or "(none)"

        prompt = WORKER_PROMPT.format(
            existing_files=existing_files_str,
            task_description=self.task.description,
            files_to_touch=", ".join(self.task.files) if self.task.files else "any",
        )

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: return raw response if LLM did not emit JSON
            return response

        files = data.get("files", {})
        summary = data.get("summary", "No summary provided.")

        written: list[str] = []
        for rel_path, content in files.items():
            await self.client.write_file(rel_path, content)
            written.append(rel_path)

        if written:
            return f"{summary}\nFiles written: {', '.join(written)}"
        return summary
