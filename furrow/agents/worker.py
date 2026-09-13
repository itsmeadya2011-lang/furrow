from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path(self.client.settings.workspace)

    async def run(self) -> str:
        logger.info("task_started", task_id=self.task.id, description=self.task.description)
        context_parts: list[str] = []
        files_to_read = self.task.files

        if files_to_read:
            for rel_path in files_to_read:
                full_path = self.workspace / rel_path
                try:
                    content = await self.client.read_file(full_path)
                    context_parts.append(f"--- File: {rel_path} ---\n{content}")
                except FileNotFoundError:
                    context_parts.append(f"--- File: {rel_path} ---\n(file does not exist yet)")
        else:
            all_files = self.client.list_files(self.workspace)
            for rel_path in all_files:
                full_path = self.workspace / rel_path
                try:
                    content = await self.client.read_file(full_path)
                    context_parts.append(f"--- File: {rel_path} ---\n{content}")
                except FileNotFoundError:
                    pass

        context = "\n\n".join(context_parts)
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"Existing file contents:\n{context}\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        parsed = self._parse_json_response(response)
        if parsed is None:
            logger.warning("task_parse_failed", task_id=self.task.id)
            return response

        files = parsed.get("files", {})
        summary = parsed.get("summary", "No summary provided")

        for path, content in files.items():
            await self.client.write_file(self.workspace / path, content)

        logger.info("task_completed", task_id=self.task.id, summary=summary)
        return summary

    @staticmethod
    def _parse_json_response(text: str) -> dict | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        try:
            start = text.index("{")
            end = text.rindex("}")
            return json.loads(text[start:end + 1])
        except (ValueError, json.JSONDecodeError):
            return None
