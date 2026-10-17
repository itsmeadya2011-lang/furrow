from __future__ import annotations

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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        # Try to parse as JSON to apply file changes
        try:
            import json

            data = json.loads(response)
            if isinstance(data, dict):
                # Apply file changes if present
                files = data.get("files", [])
                for f in files:
                    try:
                        path = f.get("path")
                        content = f.get("content")
                        if path and content is not None:
                            from pathlib import Path

                            full_path = Path(self.client.settings.workspace) / path
                            await self.client.write_file(full_path, content)
                    except Exception as e:
                        pass  # Don't fail if file write fails
                summary = data.get("summary", "")
                issues = data.get("issues", [])
                if issues:
                    return f"{summary}\nIssues: {', '.join(issues)}"
                return summary
        except Exception:
            pass
        return response
