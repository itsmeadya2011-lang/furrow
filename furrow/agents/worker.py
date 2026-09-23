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
        all_tasks: list[TaskModel] | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.all_tasks = all_tasks or []

    async def run(self) -> str:
        # Read existing files to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = ""

        # Build context from other completed tasks
        context = ""
        for t in self.all_tasks:
            if t.status == "completed" and t.result:
                context += f"\nTask {t.id} ({t.description}): {t.result[:200]}"

        files_context = ""
        if file_contents:
            files_context = "\n\nExisting files:\n" + "\n".join(
                f"--- {path} ---\n{content[:2000]}" for path, content in file_contents.items()
            )

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to modify: {', '.join(self.task.files) if self.task.files else 'any'}{context}{files_context}\n\nReturn JSON with this shape:\n{{\n  \"changes\": [\n    {{\"path\": \"src/auth.py\", \"content\": \"full file content here\"}}\n  ],\n  \"summary\": \"What was implemented\"\n}}"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            for change in changes:
                path = change.get("path")
                content = change.get("content")
                if path and content is not None:
                    await self.client.write_file(path, content)
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract file content from response
            return f"Failed to parse structured response: {e}. Raw response: {response[:500]}"
