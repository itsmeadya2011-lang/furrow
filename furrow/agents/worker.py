from __future__ import annotations

import json
from pathlib import Path

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None, workspace: Path | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())
        self.workspace = workspace

    async def run(self) -> str:
        # Build context with file operations available
        workspace_context = ""
        if self.workspace:
            files = self.client.list_files(self.workspace)
            workspace_context = "\nWorkspace files:\n" + "\n".join(f"  {f}" for f in files[:30])

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{workspace_context}\n\nAvailable tools:\n- read_file(path): Read a file\n- write_file(path, content): Write a file\n- list_files(directory): List files\n\nReturn JSON with your actions and a summary."

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Try to parse as JSON for structured actions
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "actions" in data:
                return await self._execute_actions(data["actions"])
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return the response as summary
        return response

    async def _execute_actions(self, actions: list[dict]) -> str:
        """Execute a list of file actions."""
        results = []
        for action in actions:
            action_type = action.get("type")
            if action_type == "read":
                path = action.get("path")
                if path:
                    content = await self.client.read_file(self.workspace / path if self.workspace else path)
                    results.append(f"Read {path}: {len(content)} chars")
            elif action_type == "write":
                path = action.get("path")
                content = action.get("content", "")
                if path:
                    await self.client.write_file(self.workspace / path if self.workspace else path, content)
                    results.append(f"Wrote {path}: {len(content)} chars")
            elif action_type == "list":
                directory = action.get("directory", ".")
                files = self.client.list_files(self.workspace / directory if self.workspace else directory)
                results.append(f"Listed {directory}: {files}")
        return "; ".join(results)