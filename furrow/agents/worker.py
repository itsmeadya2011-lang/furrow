from __future__ import annotations

import json

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

log = get_logger("furrow.worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

    async def run(self) -> str:
        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.settings.workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    file_contents[file_path] = content
                except Exception:  # noqa: BLE001
                    file_contents[file_path] = f"[Error reading file: {file_path}]"

        files_context = ""
        if file_contents:
            files_context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to modify: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n\nReturn your changes as JSON with this format:\n{{\n  \"changes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"new file content\"\n    }}\n  ],\n  \"summary\": \"Brief summary of changes\"\n}}"

        log.debug("worker_started", task_id=self.task.id, description=self.task.description[:100], files=self.task.files)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")

            # Apply changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content")
                if file_path and content is not None:
                    full_path = self.settings.workspace / file_path
                    await self.client.write_file(full_path, content)
                    log.info("file_written", file=file_path, size=len(content))

            log.info("worker_completed", task_id=self.task.id, summary=summary[:100], files_changed=len(changes))
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("worker_parse_failed", task_id=self.task.id, error=str(e), response=response[:200])
            # Fallback: return the raw response
            return f"Worker response (parse error: {e}): {response[:500]}"