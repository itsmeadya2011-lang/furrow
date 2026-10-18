from __future__ import annotations

import json
import os
from pathlib import Path

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import (
    FileOperation,
    Settings,
    TaskModel,
    WorkerResult,
    settings as default_settings,
)
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self) -> WorkerResult:
        workspace = str(self.settings.workspace)
        files = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            WORKER_PROMPT
            .replace("__TASK_DESCRIPTION__", self.task.description)
            .replace("__TASK_FILES__", files)
            .replace("__WORKSPACE__", workspace)
        )
        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        result = self._parse_response(response)
        await self._apply_operations(result.operations)
        return result

    def _parse_response(self, response: str) -> WorkerResult:
        try:
            data = json.loads(response)
            return WorkerResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            return WorkerResult(
                operations=[],
                summary=f"Failed to parse worker response: {e}\nResponse: {response}",
            )

    async def _apply_operations(self, operations: list[FileOperation]) -> None:
        for op in operations:
            path = self._resolve_path(op.path)
            if op.op == "write":
                if op.content is None:
                    continue
                await self.client.write_file(path, op.content)
            elif op.op == "edit":
                if op.old is None or op.new is None:
                    continue
                current = await self.client.read_file(path)
                updated = current.replace(op.old, op.new, 1)
                await self.client.write_file(path, updated)
            elif op.op == "delete":
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

    def _resolve_path(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(self.settings.workspace / p)
