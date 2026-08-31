from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


def _extract_json(response: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating surrounding text.

    Strips optional markdown code fences and returns the first JSON object found.
    """
    text = response.strip()
    if text.startswith("```"):
        # Drop the opening fence line (e.g. "```json") and the closing fence.
        lines = text.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to scanning for the first {...} block in the response.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = self.client.settings.workspace
        files_hint = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Workspace: {workspace}\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_hint}\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = _extract_json(response)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                "worker_response_parse_failed",
                task_id=self.task.id,
                error=str(e),
                response=response,
            )
            return f"Worker produced no parseable output: {e}"

        summary = str(data.get("summary", "")).strip() or "Worker completed without a summary."
        files = data.get("files") or []
        if not isinstance(files, list):
            logger.warning("worker_files_not_list", task_id=self.task.id, type=type(files).__name__)
            files = []

        written: list[str] = []
        for entry in files:
            if not isinstance(entry, dict):
                logger.warning("worker_file_entry_invalid", task_id=self.task.id, entry=entry)
                continue
            path = entry.get("path")
            content = entry.get("content", "")
            if not path or not isinstance(path, str):
                logger.warning("worker_file_path_missing", task_id=self.task.id, entry=entry)
                continue
            if not isinstance(content, str):
                logger.warning("worker_file_content_invalid", task_id=self.task.id, path=path)
                continue
            target = workspace / path
            await self.client.write_file(target, content)
            written.append(path)

        logger.info(
            "worker_completed",
            task_id=self.task.id,
            files_written=len(written),
            paths=written,
        )
        return f"{summary} (wrote {len(written)} file(s): {', '.join(written) or 'none'})"
