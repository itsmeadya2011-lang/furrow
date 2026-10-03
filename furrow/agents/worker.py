from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            payload = extract_json(response)
        except ValueError:
            log.info("worker.no_edit_block", task_id=self.task.id)
            return response
        if not isinstance(payload, dict):
            log.info("worker.edit_block_not_object", task_id=self.task.id)
            return response
        return await self._apply_edits(payload, response)

    async def _apply_edits(self, payload: dict[str, Any], raw_response: str) -> str:
        workspace = Path(self.client.settings.workspace).resolve()
        raw_edits = payload.get("edits")
        written: list[str] = []
        rejected: list[str] = []

        for edit in raw_edits if isinstance(raw_edits, list) else []:
            if not isinstance(edit, dict):
                log.warning("worker.edit_malformed", task_id=self.task.id)
                continue
            raw_path = str(edit.get("path") or "").strip()
            content = edit.get("content")
            if not raw_path or not isinstance(content, str):
                log.warning("worker.edit_malformed", task_id=self.task.id, path=raw_path)
                rejected.append(raw_path or "<missing path>")
                continue

            target = (workspace / raw_path).resolve()
            if not self._inside(target, workspace):
                log.warning("worker.edit_outside_workspace", task_id=self.task.id, path=raw_path)
                rejected.append(raw_path)
                continue

            await self.client.write_file(target, content)
            written.append(str(target.relative_to(workspace)))

        summary = str(payload.get("summary") or "").strip()
        parts = [summary] if summary else []
        if written:
            parts.append("Files written: " + ", ".join(written))
        if rejected:
            parts.append("Rejected edits outside workspace: " + ", ".join(rejected))
        if not parts:
            return raw_response
        return "\n".join(parts)

    @staticmethod
    def _inside(target: Path, workspace: Path) -> bool:
        return target == workspace or workspace in target.parents