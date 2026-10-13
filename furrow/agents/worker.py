from __future__ import annotations

import json

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except OSError:
                file_contents[file_path] = f"[File not found: {file_path}]"

        files_context = ""
        if file_contents:
            files_context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n"
        
        # The worker should return a JSON with actions to perform
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse as JSON actions, fallback to text
        try:
            actions = json.loads(response)
            return await self._execute_actions(actions)
        except json.JSONDecodeError:
            # Fallback: treat as text summary
            return response

    async def _execute_actions(self, actions: dict) -> str:
        """Execute file operations based on LLM response."""
        results = []
        
        if "read" in actions:
            for path in actions["read"]:
                try:
                    content = await self.client.read_file(path)
                    results.append(f"Read {path}: {len(content)} chars")
                except OSError as e:
                    results.append(f"Failed to read {path}: {e}")
        
        if "write" in actions:
            for item in actions["write"]:
                path = item.get("path", "")
                content = item.get("content", "")
                try:
                    await self.client.write_file(path, content)
                    results.append(f"Wrote {path}")
                except OSError as e:
                    results.append(f"Failed to write {path}: {e}")
        
        if "list" in actions:
            for path in actions["list"]:
                try:
                    files = self.client.list_files(path)
                    results.append(f"Listed {path}: {len(files)} files")
                except OSError as e:
                    results.append(f"Failed to list {path}: {e}")

        if "edit" in actions:
            for item in actions["edit"]:
                path = item.get("path", "")
                old_str = item.get("old_str", "")
                new_str = item.get("new_str", "")
                try:
                    content = await self.client.read_file(path)
                    if old_str in content:
                        new_content = content.replace(old_str, new_str)
                        await self.client.write_file(path, new_content)
                        results.append(f"Edited {path}")
                    else:
                        results.append(f"Edit failed: old_str not found in {path}")
                except OSError as e:
                    results.append(f"Failed to edit {path}: {e}")

        return "\n".join(results) if results else "No actions performed"
