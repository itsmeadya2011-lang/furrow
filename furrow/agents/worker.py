from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = self.client.settings.workspace

    async def run(self) -> str:
        # Build context with file contents if files are specified
        file_context = ""
        if self.task.files:
            file_context = "\n\nRelevant files:\n"
            for file_path in self.task.files:
                full_path = self.workspace / file_path
                if full_path.exists():
                    content = await self.client.read_file(full_path)
                    file_context += f"\n--- {file_path} ---\n{content}\n"
                else:
                    file_context += f"\n--- {file_path} ---\n[File does not exist]\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{file_context}\n"
        
        # The LLM should return a JSON with actions to perform
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse as JSON for structured actions
        try:
            actions = json.loads(response)
            if isinstance(actions, dict) and "actions" in actions:
                return await self._execute_actions(actions["actions"])
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: return the raw response as summary
        return response

    async def _execute_actions(self, actions: list[dict[str, Any]]) -> str:
        results = []
        for action in actions:
            action_type = action.get("type")
            if action_type == "read":
                path = self.workspace / action["path"]
                content = await self.client.read_file(path)
                results.append(f"Read {action['path']}: {len(content)} chars")
            elif action_type == "write":
                path = self.workspace / action["path"]
                await self.client.write_file(path, action["content"])
                results.append(f"Wrote {action['path']}")
            elif action_type == "list":
                path = self.workspace / action.get("path", ".")
                files = self.client.list_files(path)
                results.append(f"Listed {action.get('path', '.')}: {len(files)} files")
            else:
                results.append(f"Unknown action: {action_type}")
        return "\n".join(results)
