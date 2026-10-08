from __future__ import annotations

import json
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
        files_section = ""
        if self.task.files:
            snapshots = []
            for path in self.task.files:
                try:
                    content = await self.client.read_file(path)
                except OSError:
                    content = "(file does not exist yet)"
                snapshots.append(f"--- {path} ---\n{content}")
            files_section = "\nCurrent file contents:\n" + "\n".join(snapshots) + "\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{files_section}"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        data = self._parse_response(response)

        operations = data.get("operations") or []
        if not isinstance(operations, list):
            raise ValueError(f'"operations" must be a list. Response: {response}')

        written: list[str] = []
        deleted: list[str] = []
        for op in operations:
            if not isinstance(op, dict):
                raise ValueError(f"Invalid operation (not an object): {op}")
            action = op.get("action")
            path = op.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"Invalid operation (missing path): {op}")
            if action == "write":
                await self.client.write_file(path, op.get("content") or "")
                written.append(path)
            elif action == "delete":
                await self.client.delete_file(path)
                deleted.append(path)

        changes = []
        if written:
            changes.append("wrote " + ", ".join(written))
        if deleted:
            changes.append("deleted " + ", ".join(deleted))
        summary = data.get("summary") or ""
        return "; ".join(part for part in (summary, *changes) if part) or "No changes made."

    @staticmethod
    def _parse_response(response: str) -> dict:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0].strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse file operations from LLM: {e}\nResponse: {response}")
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object with operations. Response: {response}")
        return data
