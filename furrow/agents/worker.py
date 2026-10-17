from __future__ import annotations

import json
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
        files_to_touch = ", ".join(self.task.files) if self.task.files else "any"
        deps = ", ".join(self.task.dependencies) if self.task.dependencies else "none"
        prompt = WORKER_PROMPT.format(
            task_description=self.task.description,
            files_to_touch=files_to_touch,
            dependencies=deps,
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: treat the raw response as a summary.
            return response.strip()

        operations = data.get("operations", [])
        results: list[str] = []

        for op in operations:
            op_type = op.get("op", "")
            path = op.get("path", "")
            if op_type == "read":
                try:
                    content = await self.client.read_file(path)
                    results.append(f"read {path} ({len(content)} bytes)")
                except Exception as e:
                    results.append(f"read {path} FAILED: {e}")
            elif op_type == "write":
                content = op.get("content", "")
                try:
                    await self.client.write_file(path, content)
                    results.append(f"write {path} ({len(content)} bytes)")
                except Exception as e:
                    results.append(f"write {path} FAILED: {e}")

        summary = data.get("summary", "")
        issues = data.get("issues", [])
        parts: list[str] = []
        if summary:
            parts.append(summary)
        if results:
            parts.append("Operations: " + "; ".join(results))
        if issues:
            parts.append("Issues: " + "; ".join(issues))

        return " | ".join(parts) if parts else response.strip()
