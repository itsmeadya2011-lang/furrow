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
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or Path.cwd()

    async def run(self) -> str:
        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)

        context = ""
        if file_contents:
            context = "\n\nExisting files:\n" + "\n\n".join(
                f"=== {path} ===\n{content}" for path, content in file_contents.items()
            )

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Try to parse file operations from response
        await self._apply_changes(response)

        return response

    async def _apply_changes(self, response: str) -> None:
        """Parse and apply file changes from worker response.
        Expected format: JSON blocks with file operations.
        """
        try:
            # Look for JSON code blocks
            import re
            json_blocks = re.findall(r'```json\n(.*?)\n```', response, re.DOTALL)
            for block in json_blocks:
                try:
                    ops = json.loads(block)
                    if isinstance(ops, list):
                        for op in ops:
                            await self._apply_operation(op)
                    elif isinstance(ops, dict):
                        await self._apply_operation(ops)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass  # Silently ignore parsing errors

    async def _apply_operation(self, op: dict) -> None:
        action = op.get("action")
        path = op.get("path")
        if not action or not path:
            return

        full_path = self.workspace / path

        if action == "write":
            content = op.get("content", "")
            await self.client.write_file(full_path, content)
        elif action == "delete":
            if full_path.exists():
                full_path.unlink()
        elif action == "append":
            content = op.get("content", "")
            existing = await self.client.read_file(full_path) if full_path.exists() else ""
            await self.client.write_file(full_path, existing + "\n" + content)
