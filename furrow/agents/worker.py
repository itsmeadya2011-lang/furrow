from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.logger = structlog.get_logger("furrow.worker")

    async def run(self) -> str:
        self.logger.info("worker_start", task_id=self.task.id, description=self.task.description)

        # Read existing files so the LLM has context before proposing edits.
        context_parts: list[str] = []
        files = self.task.files or []
        for file_path in files:
            if await self.client.file_exists(file_path):
                content = await self.client.read_file(file_path)
                self.logger.info("file_read", file=file_path, length=len(content))
                context_parts.append(
                    f"--- BEGIN FILE: {file_path} ---\n{content}\n--- END FILE: {file_path} ---"
                )
            else:
                self.logger.info("file_not_found_skipped", file=file_path)

        files_section = ", ".join(files) if files else "any"
        context_section = (
            "\n".join(context_parts) if context_parts else "No existing files to read."
        )

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            "IMPORTANT: You must return your response as valid JSON with this exact shape:\n"
            "{\n"
            '  "edits": [\n'
            '    {"action": "write", "file": "path/to/file.py", "content": "..."},\n'
            '    {"action": "edit", "file": "path/to/file.py", '
            '"old_string": "...", "new_string": "..."}\n'
            "  ],\n"
            '  "summary": "Brief description of what was done"\n'
            "}\n\n"
            "Return ONLY this JSON (no markdown formatting, no leading text, no explanations).\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_section}\n\n"
            f"Existing file contents:\n{context_section}\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            self.logger.warning(
                "json_parse_failed", task_id=self.task.id, raw_response=response[:500]
            )
            return response

        edits = data.get("edits", [])
        applied: list[str] = []
        for edit in edits:
            try:
                action = edit["action"]
                file_path = edit["file"]
                if action == "write":
                    await self.client.write_file(file_path, edit["content"])
                    self.logger.info("edit_applied", action="write", file=file_path)
                    applied.append(f"write {file_path}")
                elif action == "edit":
                    ok = await self.client.edit_file(
                        file_path, edit["old_string"], edit["new_string"]
                    )
                    self.logger.info("edit_applied", action="edit", file=file_path, success=ok)
                    applied.append(f"edit {file_path} (matched={ok})")
                else:
                    self.logger.warning("unknown_action", action=action, file=file_path)
            except KeyError as exc:
                self.logger.warning("edit_missing_field", error=str(exc), edit=edit)

        summary = data.get("summary", "") or "Task completed."
        result = f"{summary}\nEdits applied:\n" + "\n".join(f"- {a}" for a in applied)

        self.logger.info("worker_complete", task_id=self.task.id, edits_applied=len(applied))
        return result
