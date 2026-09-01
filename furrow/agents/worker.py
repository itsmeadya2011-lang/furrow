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
        workspace: Path | str | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path(workspace) if workspace is not None else self.client.settings.workspace

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Workspace: {self.workspace}\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            "Apply any changes to disk under the workspace. Then return a JSON object "
            'with this exact shape (no markdown, no explanation):\n'
            '{"files": [{"path": "relative/path", "content": "full file content"}], '
            '"summary": "concise summary of changes"}'
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._apply(response)

    async def _apply(self, response: str) -> str:
        files_written: list[str] = []
        summary = ""
        try:
            data = json.loads(response)
            for entry in data.get("files", []):
                rel = entry["path"]
                content = entry["content"]
                target = self.workspace / rel
                await self.client.write_file(target, content)
                files_written.append(str(Path(rel)))
            summary = data.get("summary", "")
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            # LLM did not return parseable structured output; fall back to the raw text.
            summary = response
        return json.dumps({"files": files_written, "summary": summary})
