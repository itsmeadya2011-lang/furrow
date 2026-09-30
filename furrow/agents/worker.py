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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse the JSON payload returned by the LLM.
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return f"[parse_error] Could not parse LLM response as JSON. Raw response:\n{response}"

        summary = parsed.get("summary", "") if isinstance(parsed, dict) else ""
        operations = parsed.get("operations", []) if isinstance(parsed, dict) else []

        notes: list[str] = []
        for op in operations:
            action = op.get("action")
            path = op.get("path")
            if action == "write":
                content = op.get("content", "")
                try:
                    await self.client.write_file(path, content)
                    notes.append(f"Wrote {path}")
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"Failed to write {path}: {exc}")
            elif action == "read":
                try:
                    result = await self.client.read_file(path)
                    notes.append(f"Read {path} ({len(result)} chars)")
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"Failed to read {path}: {exc}")
            elif action == "list":
                try:
                    file_list = self.client.list_files(path)
                    notes.append(f"Listed {path} ({len(file_list)} files)")
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"Failed to list {path}: {exc}")
            elif action == "edit":
                old_string = op.get("old_string", "")
                new_string = op.get("new_string", "")
                try:
                    content = await self.client.read_file(path)
                    if old_string not in content:
                        notes.append(f"Failed to edit {path}: old_string not found")
                    else:
                        content = content.replace(old_string, new_string, 1)
                        await self.client.write_file(path, content)
                        notes.append(f"Edited {path}")
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"Failed to edit {path}: {exc}")
            else:
                notes.append(f"Unknown action: {action}")

        detail = "\n".join(f"- {n}" for n in notes)
        return f"{summary}\n{detail}" if detail else summary

