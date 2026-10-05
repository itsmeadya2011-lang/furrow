from __future__ import annotations

import json
import re
from pathlib import Path
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
        files = await self._read_files(self.task.files)
        prompt = self._build_prompt(files)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        operations = self._parse_response(response)
        changed = await self.execute_operations(operations.get("operations", []))
        summary = operations.get("summary", "")
        return f"Changed: {', '.join(changed)}\n\n{summary}"

    async def _read_files(self, files: list[str]) -> dict[str, str]:
        contents: dict[str, str] = {}
        for path in files:
            try:
                contents[path] = await self.client.read_file(path)
            except FileNotFoundError:
                contents[path] = "(file does not exist yet)"
        return contents

    def _build_prompt(self, files: dict[str, str]) -> str:
        file_block = ""
        for path, content in files.items():
            file_block += f"\n--- {path} ---\n{content}\n"
        return (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Current file contents:{file_block}"
        )

    def _parse_response(self, response: str) -> dict:
        match = re.search(r"\{[\s\S]*\}", response)
        if not match:
            raise ValueError(f"LLM response did not contain valid JSON: {response}")
        return json.loads(match.group(0))

    async def execute_operations(self, operations: list[dict]) -> list[str]:
        changed: list[str] = []
        for op in operations:
            action = op.get("action")
            path = op.get("path")
            if not path:
                continue
            if action == "delete":
                p = Path(path)
                if p.exists():
                    p.unlink()
                    changed.append(path)
            elif action in ("create", "update"):
                await self.client.write_file(path, op.get("content", ""))
                changed.append(path)
        return changed