from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = WORKER_PROMPT.format(
            task_description=self.task.description,
            files=", ".join(self.task.files) if self.task.files else "(any)",
        )
        response = await self.client.complete(
            prompt,
            system=WORKER_SYSTEM_PROMPT,
            model=self.client.settings.worker_model,
        )

        try:
            data = self._parse_json(response)
        except (json.JSONDecodeError, ValueError):
            if self.task.files:
                target = (self.client.settings.workspace / self.task.files[0]).resolve()
                content = f"```\n{response}\n```"
                await self.client.write_file(target, content)
                return (
                    f"Worker produced non-JSON output; saved raw response to "
                    f"{self.task.files[0]} as a code block."
                )
            raise ValueError(
                f"Worker produced non-JSON output and no target file: {response[:200]}"
            )

        summary = data.get("summary", "")
        edits = data.get("edits", [])

        workspace = self.client.settings.workspace.resolve()
        for edit in edits:
            rel_path = Path(edit["path"])
            target = (workspace / rel_path).resolve()
            if not target.is_relative_to(workspace):
                raise ValueError(
                    f"Edit path '{rel_path}' resolves outside workspace {workspace}"
                )
            await self.client.write_file(target, edit["content"])

        return f"Edited {len(edits)} file(s): {summary}"

    @staticmethod
    def _parse_json(response: str) -> dict:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
                cleaned = re.sub(r"\n```$", "", cleaned)
                return json.loads(cleaned)
            raise
