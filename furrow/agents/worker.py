from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


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
        logger.info("task_start", task_id=self.task.id)
        ws = self.client.settings.workspace
        workspace = Path(ws) if hasattr(self.client.settings, "workspace") else Path.cwd()
        files_list = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
            f"Files to touch: {files_list}\nWorkspace: {workspace}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = self._parse_json(response)
            files = data.get("files", [])
            summary = data.get("summary", "No summary provided")
            for f in files:
                try:
                    path = f.get("path") if isinstance(f, dict) else None
                    content = f.get("content") if isinstance(f, dict) else None
                    if not path:
                        continue
                    full_path = Path(path)
                    if not full_path.is_absolute():
                        full_path = workspace / full_path
                    await self.client.write_file(full_path, content or "")
                except Exception as e:  # noqa: BLE001 - intentionally broad
                    fname = f.get("path") if isinstance(f, dict) else "file"
                    summary = f"{summary}\nError writing {fname}: {e}"
            logger.info("task_complete", task_id=self.task.id, files_written=len(files))
            return summary
        except Exception as e:  # noqa: BLE001
            return response if response else f"Failed to parse worker response: {e}"

    def _parse_json(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            if "```json" in text:
                i = text.find("```json")
                j = text.find("```", i + len("```json"))
                if j != -1:
                    return json.loads(text[i + len("```json") : j].strip())
            if "```" in text:
                i = text.find("```")
                j = text.find("```", i + 3)
                if j != -1:
                    return json.loads(text[i + 3 : j].strip())
            raise
